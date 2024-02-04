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


class InteractiveFunction(InteractiveGraph):
    DEPCALC: Calculator = None
    LAYOUT: dict = {
        "ax" : [0.03, 0.25, 0.45, 0.7],
        "axdep" : [0.52, 0.25, 0.45, 0.7],
        "axslider" : [0.25, 0.13, 0.5, 0.02],
        "axcheckbox" : [0.05, 0.05, 0.075, 0.15],
        "axcheckbox2" : [0.125, 0.05, 0.075, 0.15],
        "axrandbutton" : [0.25, 0.05, 0.1, 0.05],
        "axlassobutton" : [0.45, 0.05, 0.1, 0.05],
        "axclusterbutton" : [0.65, 0.05, 0.1, 0.05],
        "axclusters" : [0.8, 0.05, 0.15, 0.15],
        "orientation" : "horizontal",
    }
    LAYOUT["cluster"] = [None,
        [(0.00, 0.67), 0.33, 0.33], [(0.33, 0.67), 0.34, 0.33], [(0.67, 0.67), 0.33, 0.33],
        [(0.00, 0.33), 0.33, 0.34], [(0.33, 0.33), 0.34, 0.34], [(0.67, 0.33), 0.33, 0.34],
        [(0.00, 0.00), 0.33, 0.33], [(0.33, 0.00), 0.34, 0.33], [(0.67, 0.00), 0.33, 0.33]
    ]
    m_dists_using: dict = {"1σ": False, "2σ": True, "3σ": False, "5σ": False, "8σ": False}
    m_dists_using_dep: dict = {"1σ": False, "2σ": True, "3σ": False, "5σ": False, "8σ": False}

    def __init__(self, data, dep_data=None, cov_data=None, mean_data=None, cov_dep=None, mean_dep=None) -> None:
        self.CALC = Calculator(data=data, cov=cov_data, cov_mean=mean_data)

        self.Palette(len(self.CALC), len(self.m_dists_using))
        self.DEPCALC = Calculator(data=dep_data, cov=cov_dep, cov_mean=mean_dep, sort = self.CALC.get_sort())

        self.CONFS: dict = {sdstr : np.sqrt(chi2.ppf((2*norm.cdf(float(sdstr[:-1])) - 1), self.CALC.get_dim())) for sdstr in self.m_dists_using.keys()}
        self.M_DISTS: list[int] = self.CONFS.values()
        self.MAX_SD = self.CALC.get_max_sds()*0.99

        self.point_colours = np.vstack([self.Palette.points_colour]*len(self.CALC))
        self.clusters = np.array([0]*len(self.CALC))
        self.num_clusters = 0
        self.lassoing = False
        
        P = P_dep = None
        if self.MAX_SD > 20:
            self.MAX_SD = 20
        for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
            sd = i*self.MAX_SD/self.PRECISION
            ind = self.CALC.partition_data(sd)
            u, v = self.CALC.optimise_plane(ind = ind, from_plane=P)
            ud, vd = self.DEPCALC.optimise_plane(ind=ind, from_plane=P_dep)
            P = np.vstack([u, v])
            P_dep = np.vstack([ud, vd])
            self.PREPLOTS = P[np.newaxis, :, :] if i==0 else np.vstack([self.PREPLOTS, P[np.newaxis, :, :]])
            self.DEPPLOTS = P_dep[np.newaxis, :, :] if i==0 else np.vstack([self.DEPPLOTS, P_dep[np.newaxis, :, :]])

        self.build_widgets()
        
        self.dep_attr_labels = []
        self.update(self.PREPLOTS[self.points_ind], self.DEPPLOTS[self.points_ind])


    def update(self, proj: ndarray | None = None, dep_proj = None):
        if proj is not None:
            self.curr_proj = proj
        if dep_proj is not None:
            self.curr_dep_proj = dep_proj

        self.ax.cla()
        self.axdep.cla()

        # points
        point_colours = self.get_point_colours()
        proj_mean = self.curr_proj @ self.CALC.get_cov_mean()
        proj_points = self.curr_proj @ self.CALC.get_data() - proj_mean
        self.curr_collection = self.ax.scatter(proj_points[0], proj_points[1], marker = ".", c=point_colours)

        proj_dep_mean = self.curr_dep_proj @ self.DEPCALC.get_cov_mean()
        proj_dep_points = self.curr_dep_proj @ self.DEPCALC.get_data() - proj_dep_mean
        self.axdep.scatter(proj_dep_points[0], proj_dep_points[1], marker = ".", c = point_colours)

        
        # ellipses
        ellipses = self.CALC.get_proj_ellipses(self.curr_proj, [self.CONFS[m] for m in self.CONFS if self.m_dists_using[m]])
        for i, ellipse in enumerate(ellipses):
            self.ax.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        dep_ellipses = self.DEPCALC.get_proj_ellipses(self.curr_dep_proj, [self.CONFS[m] for m in self.CONFS if self.m_dists_using_dep[m]])
        for i, ellipse in enumerate(dep_ellipses):
            self.axdep.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        # axes
        i = 0
        self.attr_labels.clear()
        for proj_axis in self.curr_proj.T:
            self.ax.plot([0, proj_axis[0]*self.CALC.get_max_norm()*0.5], [0, proj_axis[1]*self.CALC.get_max_norm()*0.5], c = self.Palette.axes_colour, linewidth = 1)
            new_label = self.ax.text(proj_axis[0]*self.CALC.get_max_norm()*0.5, proj_axis[1]*self.CALC.get_max_norm()*0.5, self.CALC.get_attrs()[i], picker=True)
            new_label.set_bbox(dict(facecolor=self.Palette.graph_bg, alpha=0.7, linewidth=0, boxstyle=BoxStyle.Round(pad=0.05)))
            self.attr_labels.append(new_label)
            i += 1

        i = 0
        self.dep_attr_labels.clear()
        for proj_axis in self.curr_dep_proj.T:
            self.axdep.plot([0, proj_axis[0]*self.DEPCALC.get_max_norm()*0.5], [0, proj_axis[1]*self.DEPCALC.get_max_norm()*0.5], c = self.Palette.axes_colour, linewidth = 1)
            new_label = self.axdep.text(proj_axis[0]*self.DEPCALC.get_max_norm()*0.5, proj_axis[1]*self.DEPCALC.get_max_norm()*0.5, self.DEPCALC.get_attrs()[i], picker=True)
            new_label.set_bbox(dict(facecolor=self.Palette.graph_bg, alpha=0.7, linewidth=0, boxstyle=BoxStyle.Round(pad=0.05)))
            self.dep_attr_labels.append(new_label)
            i += 1
        

        self.ax.set_xbound(-self.CALC.get_max_norm(), self.CALC.get_max_norm())
        self.ax.set_ybound(-self.CALC.get_max_norm(), self.CALC.get_max_norm())
        self.axdep.set_xbound(-self.DEPCALC.get_max_norm(), self.DEPCALC.get_max_norm())
        self.axdep.set_ybound(-self.DEPCALC.get_max_norm(), self.DEPCALC.get_max_norm())

        # Hide X and Y axes label marks
        self.axclusters.xaxis.set_tick_params(labelbottom=False)
        self.axclusters.yaxis.set_tick_params(labelleft=False)

        self.fig.canvas.draw_idle()

    def show_random(self, event):
        P = self.CALC.get_random_plane()
        P_dep = self.DEPCALC.get_random_plane()
        self.update(P, P_dep)

    def key_press(self, event):
        super().key_press(event)
        if event.key == "m":
            print(np.round(self.curr_dep_proj, 4))

    def change_cutoff(self, val=None):
        proj = super().change_cutoff(val, pres = self.PREPLOTS, update=False)
        dep_proj = super().change_cutoff(val, pres = self.DEPPLOTS, update=False)
        self.update(proj, dep_proj)

    def move_by_select(self):
        self.curr_proj = super().move_by_select(update = False)
        self.curr_dep_proj = super().move_by_select(calc = self.DEPCALC, proj = self.curr_dep_proj, update = False)
        self.lasso.disconnect_events()

    def pick_axes(self, event):
        if event.artist.axes == self.ax:
            proj = super().pick_axes(event, update = False)
            self.curr_proj = proj if proj is not None else self.curr_proj
            self.dragging_dep = False
        elif event.artist.axes == self.axdep:
            dep_proj = super().pick_axes(event, self.DEPCALC, self.curr_dep_proj, update = False)
            self.curr_dep_proj = dep_proj if dep_proj is not None else self.curr_dep_proj
            self.dragging_dep = True
        else:
            if event.mouseevent.button is MouseButton.RIGHT:
                self.curr_proj = super().pick_axes(event, update=False)
                self.curr_dep_proj = super().pick_axes(event, calc=self.DEPCALC, proj = self.curr_dep_proj, update=False)
            else:
                super().pick_axes(event)

        self.update()
    
    def drag_axes(self, event):
        if self.dragging_dep:
            self.curr_dep_proj = super().drag_axes(event, proj = self.curr_dep_proj, calc = self.DEPCALC, update = False)
        else:
            self.curr_proj = super().drag_axes(event, update = False)
        self.update()

    def show_ellipse_dep(self, label):
        self.m_dists_using_dep[label] = not self.m_dists_using_dep[label]
        self.update()

    def build_widgets(self):
        super().build_widgets(ellipses=False)
        self.axdep = self.fig.add_axes(self.LAYOUT["axdep"], facecolor = self.Palette.graph_bg)
        self.axcheckbox2 = self.fig.add_axes(self.LAYOUT["axcheckbox2"], facecolor = self.Palette.slider_bg)
        self.axdep.set_aspect('equal', adjustable='box')

        self.Palette.remove_border(self.axcheckbox)
        self.Palette.remove_border(self.axcheckbox2)
        #self.axcheckbox.spines['right'].set_color(self.Palette.slider_bg)
        #self.axcheckbox2.spines['left'].set_color(self.Palette.slider_bg)
        self.m_checkbox = CheckButtons(
            ax = self.axcheckbox,
            labels = list(self.CONFS.keys()),
            label_props={'color': self.Palette.ellipse_colours, "size":[14]*len(self.CONFS), "family":['serif']*len(self.CONFS)},
            frame_props={'edgecolor': self.Palette.ellipse_colours, "facecolor":'white'},
            check_props={'facecolor': self.Palette.ellipse_colours},
            actives=list(self.m_dists_using.values())
        )
        self.m_checkbox.on_clicked(self.show_ellipse)
        self.m_checkbox2 = CheckButtons(
            ax = self.axcheckbox2,
            labels = list(self.CONFS.keys()),
            label_props={'color': self.Palette.ellipse_colours, "size":[14]*len(self.CONFS), "family":['serif']*len(self.CONFS)},
            frame_props={'edgecolor': self.Palette.ellipse_colours, "facecolor":'white'},
            check_props={'facecolor': self.Palette.ellipse_colours},
            actives=list(self.m_dists_using_dep.values())
        )
        self.m_checkbox2.on_clicked(self.show_ellipse_dep)


        self.axclusters.set_title("")
        self.axslider.set_title("standard deviations")



'''
# Save to binary file
if __name__=='__main__':
    #this_graph = InteractiveGraph(data="samples/p5pexample/np.csv", cov_data="samples/p5pexample/cov_mat.csv", mean_data="samples/p5pexample/centre_of_ellipses.csv")
    
    ifunc = InteractiveFunction(dep_data="samples/bphys/testinput.csv", data="samples/bphys/testoutput.csv", cov_data = "samples/bphys/cov.csv")
    #writefile = input("Enter path to save widget (.pickle extension): ")
    #pl.dump((this_graph.PREPLOTS, this_graph.CALC, this_graph.limits), open(writefile,'wb'))
    plt.show()
'''