import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from numpy import ndarray
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, chi2
from calculator import Calculator
from interactive_graph import InteractiveGraph
from matplotlib.widgets import CheckButtons
from matplotlib.patches import BoxStyle
import warnings
warnings.filterwarnings("error")

# Including a second dataset
class InteractiveFunction(InteractiveGraph):

    """
        Stores and manages two sets of data.
    """

    # constant
    DEPCALC: Calculator = None
    DEPCONFS: dict = None
    LAYOUT: dict = {
        "ax" : [0.03, 0.25, 0.45, 0.7],
        "axdep" : [0.52, 0.25, 0.45, 0.7],
        "axslider" : [0.25, 0.13, 0.5, 0.02],
        "axcheckbox" : [0.05, 0.05, 0.075, 0.15],
        "axcheckbox2" : [0.125, 0.05, 0.075, 0.15],
        "axinfobutton" : [0.25, 0.05, 0.1, 0.05],
        "axlassobutton" : [0.45, 0.05, 0.1, 0.05],
        "axclusterbutton" : [0.65, 0.05, 0.1, 0.05],
        "axclusters" : [0.8, 0.05, 0.15, 0.15],
        "orientation" : "horizontal",
        "cluster" : [None,
            [(0.00, 0.67), 0.33, 0.33], [(0.33, 0.67), 0.34, 0.33], [(0.67, 0.67), 0.33, 0.33],
            [(0.00, 0.33), 0.33, 0.34], [(0.33, 0.33), 0.34, 0.34], [(0.67, 0.33), 0.33, 0.34],
            [(0.00, 0.00), 0.33, 0.33], [(0.33, 0.00), 0.34, 0.33], [(0.67, 0.00), 0.33, 0.33]
        ]
    }

    # variable
    m_dists_using_dep: dict = {"1σ": False, "2σ": True, "3σ": False, "5σ": False, "8σ": False}
    ellipse_with_slider_dep = False
    dep_attr_labels: list = []
    axdeplength: float = None
    xd0 = xd1 = yd0 = yd1 = None
    dragging_dep: bool = None

    # widgets
    axdep: plt.Axes = None
    axcheckbox2: plt.Axes = None
    m_checkbox2: CheckButtons = None

    
    def __init__(self, data, dep_data=None, cov_data=None, mean_data=None, cov_dep=None, mean_dep=None, scale = False, scale_dep = False) -> None:
        """Initialise interactive function"""
        super().__init__(data, cov_data, mean_data, update = False, scale=scale)
        self.DEPCALC = Calculator(data=dep_data, cov=cov_dep, cov_mean=mean_dep, sort = self.CALC.get_sort(), scale=scale_dep)
        self.DEPCONFS: dict = {sdstr : np.sqrt(chi2.isf(2*norm.sf(float(sdstr[:-1])), self.DEPCALC.get_dim())) for sdstr in self.m_dists_using.keys()}
        
        self.build_preplots()
        self.build_widgets()
        self.update(self.PREPLOTS[self.points_ind], self.DEPPLOTS[self.points_ind])


    def build_preplots(self):
        """Build preplots to be referenced later. Called in __init__"""
        P = P_dep = None
        for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
            sd = i*self.MAX_SD/self.PRECISION
            ind = self.CALC.partition_data(sd)
            
            P = self.CALC.optimise_plane(ind = ind, from_plane=P)
            P_dep = self.DEPCALC.optimise_plane(ind=ind, from_plane=P_dep)
            
            self.PREPLOTS = P[np.newaxis, :, :] if i==0 else np.vstack([self.PREPLOTS, P[np.newaxis, :, :]])
            self.DEPPLOTS = P_dep[np.newaxis, :, :] if i==0 else np.vstack([self.DEPPLOTS, P_dep[np.newaxis, :, :]])
    

    def update(self, proj: ndarray | None = None, dep_proj = None):
        """
        Re-plot everything. Called when:
        - changing projection
        - changing point colours
        - resizing box
        - changing ellipses
        """
        super().update(proj)

        if dep_proj is not None:
            self.curr_dep_proj = dep_proj

        self.axdep.cla()

        # points
        point_colours = self.get_point_colours()
        proj_dep_mean = self.curr_dep_proj @ self.DEPCALC.get_cov_mean()
        proj_dep_points = self.curr_dep_proj @ self.DEPCALC.get_data() - proj_dep_mean
        self.axdep.scatter(proj_dep_points[0], proj_dep_points[1], marker = ".", c = point_colours)


        # ellipses
        if self.ellipse_with_slider_dep:
            sds = self.cutoff_slider.val
            m_dist = np.sqrt(chi2.isf(2*norm.sf(float(sds)), self.DEPCALC.get_dim()))
            ellipse = self.DEPCALC.get_proj_ellipses(self.curr_dep_proj, [m_dist])[0]
            self.axdep.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[-1], linewidth = 2)

        else:
            dep_ellipses = self.DEPCALC.get_proj_ellipses(self.curr_dep_proj, [self.DEPCONFS[m] for m in self.DEPCONFS if self.m_dists_using_dep[m]])
            elcolours = [i for i, v in enumerate(self.m_dists_using_dep.values()) if v]
            for i, ellipse in zip(elcolours, dep_ellipses):
                self.axdep.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        # axes
        i = 0
        self.dep_attr_labels.clear()
        if self.xd0*self.xd1 < 0 and self.yd0*self.yd1 < 0:  # origin is in frame
            for proj_axis in self.curr_dep_proj.T*self.axdeplength:
                self.axdep.plot([0, proj_axis[0]], [0, proj_axis[1]], c = self.Palette.axes_colour, linewidth = 1)
                
                # axis label
                new_label = self.axdep.text(proj_axis[0], proj_axis[1], self.DEPCALC.get_attrs()[i], picker=True)
                new_label.set_bbox(dict(facecolor=self.Palette.graph_bg, alpha=0.7, linewidth=0, boxstyle=BoxStyle.Round(pad=0.05)))
                self.dep_attr_labels.append(new_label)
                i += 1
        
        # bounds
        self.axdep.set_xbound(self.xd0, self.xd1)
        self.axdep.set_ybound(self.yd0, self.yd1)

        self.axdep.callbacks.connect("xlim_changed", self.zoomed_dep)
        self.axdep.callbacks.connect("ylim_changed", self.zoomed_dep)

        self.fig.canvas.draw_idle()


    def zoomed_dep(self, event_ax):
        """Update plot boundaries. Called on pan/zoom"""
        self.xd0, self.xd1 = event_ax.get_xlim()
        self.yd0, self.yd1 = event_ax.get_ylim()
        self.axdeplength = min(self.xd1-self.xd0, self.yd1-self.yd0)*0.25
        self.update()


    def show_random(self):
        """Show random projections, call when 'R' pressed"""
        P = self.CALC.get_random_plane()
        P_dep = self.DEPCALC.get_random_plane()
        self.update(P, P_dep)


    def key_press(self, event):
        """Manages all key presses. Mostly handled in parent class"""
        super().key_press(event)
        if event.key == "m":  # print second projection
            print(np.round(self.curr_dep_proj, 4))


    def change_cutoff(self, val=None):
        """Moves projections to optimal based on slider cutoff. Called when slider is changed"""
        proj = super().change_cutoff(val, pres = self.PREPLOTS, update=False)
        dep_proj = super().change_cutoff(val, pres = self.DEPPLOTS, update=False)
        self.update(proj, dep_proj)


    def move_by_select(self):
        """Changes projection to best show lassoed points, called when points lassoed and 'enter' pressed."""
        self.curr_proj = super().move_by_select(update = False)
        self.curr_dep_proj = super().move_by_select(calc = self.DEPCALC, proj = self.curr_dep_proj, update = False)
        self.lasso.disconnect_events()


    def pick_axes(self, event):
        """Called when a pickable object is selected. This includes cluster rectangles and axes."""
        if event.artist.axes == self.ax:  # move independent axes
            proj = super().pick_axes(event, update = False)
            self.curr_proj = proj if proj is not None else self.curr_proj
            self.dragging_dep = False
        elif event.artist.axes == self.axdep:  # move dependent axes
            dep_proj = super().pick_axes(event, self.DEPCALC, self.curr_dep_proj, update = False)
            self.curr_dep_proj = dep_proj if dep_proj is not None else self.curr_dep_proj
            self.dragging_dep = True
        else:
            if event.mouseevent.button is MouseButton.RIGHT:  # select cluster
                self.curr_proj = super().pick_axes(event, update=False)
                self.curr_dep_proj = super().pick_axes(event, calc=self.DEPCALC, proj = self.curr_dep_proj, update=False)
            else:  # delete cluster
                super().pick_axes(event)

        self.update()

    
    def drag_axes(self, event):
        """Used to drag axes around, called when axis picked."""
        if self.dragging_dep:
            self.curr_dep_proj = super().drag_axes(event, proj = self.curr_dep_proj, axlength = self.axdeplength, update = False)
        else:
            self.curr_proj = super().drag_axes(event, update = False)
        self.update()


    def show_ellipse_dep(self, label):
        """Toggles an ellipse on the right. Called when checkbox ticked or unticked."""
        if label in self.m_dists_using_dep:
            self.m_dists_using_dep[label] = not self.m_dists_using_dep[label]
        else:  # toggle self.ellipse_with_slider_dep
            self.ellipse_with_slider_dep = not self.ellipse_with_slider_dep
        self.update()


    def build_widgets(self):
        """Builds and modifies widgets from parent class"""
        super().build_widgets(ellipses=False)
        
        # additional axes
        self.axdep = self.fig.add_axes(self.LAYOUT["axdep"], facecolor = self.Palette.graph_bg)
        self.axcheckbox2 = self.fig.add_axes(self.LAYOUT["axcheckbox2"], facecolor = self.Palette.slider_bg)
        
        # adjust dependent plot limits
        self.axdep.set_aspect('equal', adjustable='box')
        self.xd0 = self.yd0 = -self.DEPCALC.get_max_norm()
        self.xd1 = self.yd1 = self.DEPCALC.get_max_norm()
        self.axdeplength = 0.5*self.DEPCALC.get_max_norm()

        # checkboxes
        self.Palette.remove_border(self.axcheckbox)
        self.Palette.remove_border(self.axcheckbox2)
        self.m_checkbox = CheckButtons(
            ax = self.axcheckbox,
            labels = list(self.CONFS.keys()) + ["slider"],
            label_props={'color': self.Palette.ellipse_colours, "size":[14]*(len(self.CONFS)+1), "family":['serif']*(len(self.CONFS)+1)},
            frame_props={'edgecolor': self.Palette.ellipse_colours, "facecolor":'white'},
            check_props={'facecolor': self.Palette.ellipse_colours},
            actives=list(self.m_dists_using.values()) + [False]
        )
        self.m_checkbox.on_clicked(self.show_ellipse)
        self.m_checkbox2 = CheckButtons(
            ax = self.axcheckbox2,
            labels = list(self.DEPCONFS.keys()) + ["slider"],
            label_props={'color': self.Palette.ellipse_colours, "size":[14]*(len(self.DEPCONFS)+1), "family":['serif']*(len(self.DEPCONFS)+1)},
            frame_props={'edgecolor': self.Palette.ellipse_colours, "facecolor":'white'},
            check_props={'facecolor': self.Palette.ellipse_colours},
            actives=list(self.m_dists_using_dep.values()) + [False]
        )
        self.m_checkbox2.on_clicked(self.show_ellipse_dep)


        # adjust titles
        self.axclusters.set_title("")
        self.axslider.set_title("standard deviations")
