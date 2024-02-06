import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, chi2
from calculator import Calculator
from matplotlib.widgets import Button, Slider, CheckButtons, LassoSelector
from matplotlib.patches import BoxStyle, Rectangle
from matplotlib.path import Path
import warnings
warnings.filterwarnings("error")


# Keeping track of interactive elements
class InteractiveGraph:

    """
        Class storing references to every element of the matplotlib data visualisation widget.
    """

    # Colours
    class Palette:
        """
            Stores fonts and colours for InteractiveGraph and InteractiveFunction.
        """

        light_green = np.array([0.898, 0.988, 0.761])
        green = np.array([0.616, 0.878, 0.678])
        blue = np.array([0.271, 0.678, 0.659])
        dark_blue = np.array([0.329, 0.475, 0.502])
        grey = np.array([0.349, 0.31, 0.31])
        off_white = np.array([0.969, 0.945, 0.929])
        black = np.array([0, 0, 0])

        # colours in use
        suggest_colour = np.array([0.329, 0.475, 0.502, 1])
        ellipse_colour = blue
        bg_colour = green
        slider_colour = dark_blue
        slider_bg = np.array([0.969, 0.945, 0.929, 0.5])
        graph_bg = off_white
        axes_colour = black
        cluster_colours = np.array([[80, 80, 80, 255],
            [255, 84, 0, 255], [255, 142, 0, 255], [255, 210, 0, 255],
            [129, 230, 80, 255], [0, 210, 103, 255], [0, 192, 255, 255],
            [139, 72, 254, 255], [202, 65, 252, 255], [255, 70, 251, 255]
        ])/255


        title_font = {"color": off_white, "family": "serif"}
        subtitle_font = {"color": off_white, "family": "serif", "size": 14}

        @classmethod
        def __init__(cls, num_points, num_ellipses) -> None:
            """Sets ellipse colours and redundant points alpha value, which depends on IG attributes."""
            cls.ellipse_colours = np.outer(np.sqrt(np.sqrt(np.reciprocal(np.array(range(1, num_ellipses+1)).astype(float)))), cls.ellipse_colour)
            alphas = (np.ones((10, 1)))/np.sqrt(num_points)
            cls.cluster_colours_light = np.hstack([cls.cluster_colours[:, :3], alphas])
        
        @classmethod
        def remove_border(cls, ax: plt.Axes) -> None:
            """Sets border of an axes to the background colour."""
            ax.spines['bottom'].set_color(cls.bg_colour)
            ax.spines['top'].set_color(cls.bg_colour)
            ax.spines['right'].set_color(cls.bg_colour)
            ax.spines['left'].set_color(cls.bg_colour)

    # constants
    MAX_SD: float = None
    PRECISION: int = 20
    PREPLOTS: np.ndarray = None
    CALC: Calculator = None
    CONFS: dict = None
    LAYOUT: dict = {
        "ax" : [0.35, 0.1, 0.6, 0.8],
        "axdep" : None,
        "axslider" : [0.24, 0.1, 0.02, 0.8],
        "axcheckbox" : [0.1, 0.65, 0.1, 0.25],
        "axinfobutton" : [0.1, 0.58, 0.1, 0.05],
        "axlassobutton" : [0.1, 0.51, 0.1, 0.05],
        "axclusterbutton" : [0.1, 0.44, 0.1, 0.05],
        "axclusters" : [0.1, 0.1, 0.1, 0.3],
        "orientation" : "vertical",
        "cluster" : [None,
            [(0.00, 0.67), 0.33, 0.33], [(0.33, 0.67), 0.34, 0.33], [(0.67, 0.67), 0.33, 0.33],
            [(0.00, 0.33), 0.33, 0.34], [(0.33, 0.33), 0.34, 0.34], [(0.67, 0.33), 0.33, 0.34],
            [(0.00, 0.00), 0.33, 0.33], [(0.33, 0.00), 0.34, 0.33], [(0.67, 0.00), 0.33, 0.33]
        ]
    }

    # variable
    axlength: int = None
    curr_proj: np.ndarray = None
    points_ind: int | np.ndarray = 0
    curr_collection = None
    dragged: int = None
    clusters: np.ndarray = None
    clusters_in_use: list[int] = [0]
    num_clusters: int = 0
    m_dists_using: dict = {"1σ": False, "2σ": True, "3σ": False, "5σ": False, "8σ": False}
    attr_labels: list = []
    suggest_ind: np.ndarray = None
    lasso = None
    lassoing: bool = False
    x0 = x1 = y0 = y1 = None
    drag_id = None
    release_id = None

    # widgets
    fig = None
    
    ax: plt.Axes = None
    axslider: plt.Axes = None
    axcheckbox: plt.Axes = None
    axinfobutton: plt.Axes = None
    axlassobutton: plt.Axes = None
    axclusterbutton: plt.Axes = None
    axclusters: plt.Axes = None

    cutoff_slider: Slider = None
    m_checkbox: CheckButtons = None
    info_button: Button = None
    lasso_button: Button = None
    cluster_button: Button = None


    def __init__(self, data, cov_data = None, mean_data = None, update = True) -> None:
        """Initialise, building preplots, widgets, Palette and Calculator"""
        self.CALC = Calculator(data=data, cov=cov_data, cov_mean=mean_data)
        self.CONFS: dict = {sdstr : np.sqrt(chi2.ppf((2*norm.cdf(float(sdstr[:-1])) - 1), self.CALC.get_dim())) for sdstr in self.m_dists_using.keys()}
        self.MAX_SD = self.CALC.get_max_sds()*0.99
        if self.MAX_SD > 40:
            self.MAX_SD = 40

        self.Palette(len(self.CALC), len(self.m_dists_using))

        self.clusters = np.array([0]*len(self.CALC))

        if update:  # helps connect to InteractiveFunction class
            self.build_preplots()
            self.build_widgets()
            self.update(self.PREPLOTS[self.points_ind])


    def build_preplots(self):
        """Build preplots to be referenced later. Called in __init__"""
        P = None
        for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
            sd = i*self.MAX_SD/self.PRECISION
            ind = self.CALC.partition_data(sd)
            P = self.CALC.optimise_plane(ind=ind, from_plane=P)
            self.PREPLOTS = P[np.newaxis, :, :] if i==0 else np.vstack([self.PREPLOTS, P[np.newaxis, :, :]])

    
    def update(self, proj: np.ndarray | None = None):
        """
        Re-plot everything. Called when:
        - changing projection
        - changing point colours
        - resizing box
        - changing ellipses
        """

        if proj is not None:
            self.curr_proj = proj

        self.ax.cla()

        # points
        point_colours = self.get_point_colours()
        proj_mean = self.curr_proj @ self.CALC.get_cov_mean()
        proj_points = self.curr_proj @ self.CALC.get_data() - proj_mean  # centering
        self.curr_collection = self.ax.scatter(proj_points[0], proj_points[1], c = point_colours, marker = ".")
        
        # ellipses
        ellipses = self.CALC.get_proj_ellipses(self.curr_proj, [self.CONFS[m] for m in self.CONFS if self.m_dists_using[m]])
        elcolours = [i for i, v in enumerate(self.m_dists_using.values()) if v]  # getting correct colour index
        for i, ellipse in zip(elcolours, ellipses):
            self.ax.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        # axes
        i = 0
        self.attr_labels.clear()
        if self.x0*self.x1 < 0 and self.y0*self.y1 < 0:  # check origin is in frame
            for proj_axis in self.curr_proj.T*self.axlength:
                self.ax.plot([0, proj_axis[0]], [0, proj_axis[1]], c = self.Palette.axes_colour, linewidth = 1)

                # draw label
                new_label = self.ax.text(proj_axis[0], proj_axis[1], self.CALC.get_attrs()[i], picker=True)
                new_label.set_bbox(dict(facecolor=self.Palette.graph_bg, alpha=0.7, linewidth=0, boxstyle=BoxStyle.Round(pad=0.05)))
                self.attr_labels.append(new_label)
                i += 1
        
        # bounds
        self.ax.set_xbound(self.x0, self.x1)
        self.ax.set_ybound(self.y0, self.y1)

        self.ax.callbacks.connect("xlim_changed", self.zoomed)
        self.ax.callbacks.connect("ylim_changed", self.zoomed)

        self.fig.canvas.draw_idle()


    def get_point_colours(self) -> np.ndarray:
        """Build num_points x 4 array of point colours based on selected, cluster, suggest"""
        point_colours = self.Palette.cluster_colours_light[self.clusters]
        if type(self.points_ind) is np.ndarray:
            point_colours[self.points_ind] = self.Palette.cluster_colours[self.clusters[self.points_ind]]
        else:
            point_colours[self.points_ind:] = self.Palette.cluster_colours[self.clusters[self.points_ind:]]
        if self.suggest_ind is not None:
            point_colours[self.suggest_ind] = self.Palette.suggest_colour
        return point_colours


    def zoomed(self, event_ax):
        """Update plot boundaries. Called on pan/zoom"""
        self.x0, self.x1 = event_ax.get_xlim()
        self.y0, self.y1 = event_ax.get_ylim()
        self.axlength = min(self.x1-self.x0, self.y1-self.y0)*0.25
        self.update()


    def pick_axes(self, event, calc = None, proj = None, update = True):
        """Called when a pickable object is selected. This includes cluster rectangles and axes."""
        # InteractiveFunction compatability
        if calc is None:
            calc = self.CALC
        if proj is None:
            proj = self.curr_proj

        # cluster rectangle selected
        if isinstance(event.artist, Rectangle):
            this_cluster = int(event.artist.get_gid())
            
            if event.mouseevent.dblclick:  # delete cluster
                self.clusters_in_use.remove(this_cluster)
                self.clusters[np.where(self.clusters == this_cluster)[0]] = 0
                self.num_clusters -= 1
                event.artist.remove()
                self.update()
            
            elif event.mouseevent.button is MouseButton.RIGHT:  # select cluster
                self.points_ind = np.where(self.clusters == this_cluster)[0]
                new_proj = calc.optimise_plane(ind = self.points_ind, from_plane=proj)
                if update:
                    self.update(new_proj)
                else:
                    return new_proj
        
        # axes selected
        elif event.mouseevent.dblclick:  # send axes to 0
            ind = np.where(calc.get_attrs() == event.artist.get_text())[0][0]
            u, v = proj[0], proj[1]
            u[ind] = v[ind] = 0
            try:
                u, v = self.CALC.orthonormalise(u, v)
            except RuntimeWarning:
                pass
            else:
                if update:
                    self.update(np.vstack([u, v]))
                else:
                    return np.vstack([u, v])
        
        else:  # dragging axes
            self.dragged = np.where(calc.get_attrs() == event.artist.get_text())[0][0]
            self.drag_id = self.fig.canvas.mpl_connect("motion_notify_event", self.drag_axes)
            self.release_id = self.fig.canvas.mpl_connect('button_release_event', self.stop_dragging)
            

    def drag_axes(self, event, proj = None, axlength = None, update = True):
        """Used to drag axes around, called when axis picked."""
        if proj is None:
            proj = self.curr_proj
        if axlength is None:
            axlength = self.axlength

        try:
            pos = np.array([event.xdata/axlength, event.ydata/axlength])
        except:
            self.stop_dragging()
            return
        
        new_proj = self.CALC.move_axis(proj, self.dragged, pos)
        if update:
            self.update(new_proj)
        else:
            return new_proj


    def stop_dragging(self, event = None):
        """Releases axis being dragged, called in drag_axes"""
        self.dragged = None
        self.fig.canvas.mpl_disconnect(self.drag_id)
        self.fig.canvas.mpl_disconnect(self.release_id)


    def change_cutoff(self, val = None, pres = None, update = True):
        """Moves projection to optimal based on slider cutoff. Called when slider is changed"""
        if pres is None:  # for InteractiveFunction
            pres = self.PREPLOTS
        
        proj_ind = self.cutoff_slider.val*self.PRECISION/self.MAX_SD
        if int(proj_ind) >= self.PRECISION - 1:  # shouldn't happen, but just in case
            proj = pres[self.PRECISION-1]
        else:  # find projection between two projections
            proj_low = pres[int(proj_ind)]
            proj_high = pres[int(proj_ind) + 1]
            frac = proj_ind - int(proj_ind)
            proj = proj_low + frac*(proj_high - proj_low)
        
        # change selected points
        self.points_ind = self.CALC.partition_data(self.cutoff_slider.val)
        
        if update:
            self.update(proj)
        else:
            return proj


    def show_random(self):
        """Show random projection, called when 'R' pressed"""
        rand_proj = self.CALC.get_random_plane()
        self.update(rand_proj)


    def lasso_select(self, event):
        """Initialise lasso. Called when lasso button pressed."""
        self.lasso_button.color = self.Palette.blue
        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.lassoing = True


    def on_select(self, verts):
        """Colour lassoed points, called when lasso path is drawn."""
        path = Path(verts)
        self.suggest_ind = np.nonzero(path.contains_points(self.curr_collection.get_offsets()))[0]
        
        self.update()
        self.ax.set_title("Press enter to change projection, Y to add cluster")
        self.fig.canvas.draw_idle()


    def move_by_select(self, calc = None, proj = None, update = True):
        """Changes projection to best show lassoed points, called when points lassoed and 'enter' pressed."""
        # InteractiveFunction compatibility
        if calc is None:
            calc = self.CALC
        if proj is None:
            proj = self.curr_proj
        
        # select suggested points
        if self.suggest_ind is not None:
            self.points_ind = self.suggest_ind
            self.suggest_ind = None

        # change projection
        self.ax.set_title("Loading projection...")
        self.fig.canvas.draw_idle()
        new_proj = calc.optimise_plane(ind = self.points_ind, from_plane=proj)
        
        if update:
            self.curr_proj = new_proj
            self.lasso.disconnect_events()
        else:
            return new_proj


    def add_cluster(self):
        """Adds a new cluster. Called when points lassoed and 'Y' pressed."""
        if self.num_clusters == 9:  # no room for more clusters
            self.lasso.disconnect_events()
            self.suggest_ind = None
            return
        
        # Update internal cluster storage
        self.num_clusters += 1
        this_cluster = [i for i in range(10) if i not in self.clusters_in_use][0]
        self.clusters_in_use.append(this_cluster)
        self.clusters[self.suggest_ind] = this_cluster
        self.suggest_ind = None
        
        # Draw rectangle
        self.axclusters.add_patch(Rectangle(
            self.LAYOUT["cluster"][this_cluster][0], 
            self.LAYOUT["cluster"][this_cluster][1],
            self.LAYOUT["cluster"][this_cluster][2],
            color = self.Palette.cluster_colours[this_cluster],
            linewidth = 0,
            picker = True,
            gid = str(this_cluster)
        ))
        self.lasso.disconnect_events()


    def clear_clusters(self):
        """Reset clustering. Called when 'z' pressed."""
        # Reset internals
        self.clusters = np.array([0]*len(self.CALC))
        self.num_clusters = 0
        self.clusters_in_use = [0]
        
        self.axclusters.cla()  # Remove all rectangles
        self.axclusters.set_xticks([])
        self.axclusters.set_yticks([])
        
        self.update()


    def auto_cluster(self, event = None):
        """Use an algorithm to cluster the data. Called when auto cluster button pressed."""
        self.clear_clusters()
        self.ax.set_title("Clustering...")
        self.fig.canvas.draw_idle()
        
        self.num_clusters, new_clusters = self.CALC.get_clusters(self.points_ind)

        # set internal clusters
        if type(self.points_ind) is np.ndarray:
            self.clusters[self.points_ind] = new_clusters
        else:
            self.clusters[self.points_ind:] = new_clusters
        
        # Draw rectangles
        for i in range(1, min(self.num_clusters + 1, 10)):
            self.clusters_in_use.append(i)
            self.axclusters.add_patch(Rectangle(
                self.LAYOUT["cluster"][i][0], 
                self.LAYOUT["cluster"][i][1],
                self.LAYOUT["cluster"][i][2],
                color = self.Palette.cluster_colours[i],
                linewidth = 0,
                picker = True,
                gid = str(i)
            ))
        
        self.update()


    def print_info(self, event = None):
        """Print controls to the terminal. Called when print info button pressed."""
        print("\n~~ CONTROLS ~~")
        print("Key presses")
        print("R: Show random projection")
        print("Z: Clear all clusters")
        print("N: Print CSV file line numbers of selected points")
        print("M: Print the current projection matrix")
        print("\nCluster controls")
        print("Right click: Select points in cluster and show optimal projection")
        print("Double click: Remove cluster")
        print("\nLasso controls")
        print("Press Y: Lasso selection becomes new cluster")
        print("Press Enter: Select lassoed points and show optimal projection")
        print("~~~")


    def key_press(self, event):
        """Manages all key presses."""
        if self.lassoing:
            self.lassoing = False
            self.lasso_button.color = self.Palette.slider_colour
            
            if event.key == "enter":  # optimise projection
                self.move_by_select()
            elif event.key == "y":  # add cluster
                self.add_cluster()
            else:  # escape lassoing
                self.suggest_ind = None
                self.lasso.disconnect_events()
            self.update()

        elif event.key == "r":  # show random projection
            self.show_random(event)

        elif event.key == "z":  # clear clusters
            self.clear_clusters()

        elif event.key == "m":  # print projection
            print("\n~~ SELECTED PROJECTION ~~")
            print(np.round(self.curr_proj, 4))

        elif event.key == "n":  # print point ids
            ids = self.CALC.get_sort()[self.points_ind] if type(self.points_ind) is np.ndarray else self.CALC.get_sort()[self.points_ind:]
            print("\n~~ SELECTED POINT IDS ~~")
            print(np.sort(ids) + self.CALC.csv_dist)


    def show_ellipse(self, label):
        """Toggles an ellipse. Called when checkbox ticked or unticked."""
        self.m_dists_using[label] = not self.m_dists_using[label]
        self.update()


    def build_widgets(self, ellipses = True):
        """Builds elements of app. Called once, on initialisation."""
        
        # figure and axes
        self.fig = plt.figure(facecolor=self.Palette.bg_colour)
        self.ax = self.fig.add_axes(self.LAYOUT["ax"], facecolor = self.Palette.graph_bg)
        self.axslider = self.fig.add_axes(self.LAYOUT["axslider"], facecolor=self.Palette.slider_bg)
        self.axcheckbox = self.fig.add_axes(self.LAYOUT["axcheckbox"], facecolor=self.Palette.slider_bg)
        self.axinfobutton = self.fig.add_axes(self.LAYOUT["axinfobutton"])
        self.axlassobutton = self.fig.add_axes(self.LAYOUT["axlassobutton"])
        self.axclusterbutton = self.fig.add_axes(self.LAYOUT["axclusterbutton"])
        self.axclusters = self.fig.add_axes(self.LAYOUT["axclusters"], facecolor=self.Palette.slider_bg)

        # set plot limits
        self.ax.set_aspect('equal', adjustable='box')
        self.x0 = self.y0 = -self.CALC.get_max_norm()
        self.x1 = self.y1 = self.CALC.get_max_norm()
        self.axlength = 0.5*self.CALC.get_max_norm()

        # Slider

        self.axslider.set_title("Standard \ndeviations", fontdict=self.Palette.subtitle_font)
        self.cutoff_slider = Slider(
            ax=self.axslider,
            label='',
            valmin=0,
            valmax=self.MAX_SD,
            valinit=0,
            handle_style={"edgecolor":self.Palette.slider_colour, "facecolor": self.Palette.graph_bg},
            orientation=self.LAYOUT["orientation"],
            color=self.Palette.slider_colour,
            track_color=self.Palette.slider_bg,
            closedmax=False,
            initcolor=None
        )
        self.cutoff_slider.on_changed(self.change_cutoff)

        # Checkboxes

        if ellipses:
            self.axcheckbox.set_title("Ellipses", fontdict=self.Palette.subtitle_font)
            self.Palette.remove_border(self.axcheckbox)

            self.m_checkbox = CheckButtons(
                ax = self.axcheckbox,
                labels = self.CONFS.keys(),
                label_props={'color': self.Palette.ellipse_colours, "size":[14]*len(self.CONFS), "family":['serif']*len(self.CONFS)},
                frame_props={'edgecolor': self.Palette.ellipse_colours, "facecolor":'white'},
                check_props={'facecolor': self.Palette.ellipse_colours},
                actives=self.m_dists_using.values()
            )
            self.m_checkbox.on_clicked(self.show_ellipse)


        # Info Button
            
        self.Palette.remove_border(self.axinfobutton)

        self.info_button = Button(
            ax=self.axinfobutton,
            label = "Print help",
            color = self.Palette.slider_colour,
            hovercolor = self.Palette.blue
        )

        self.info_button.label.set_color(self.Palette.off_white)
        self.info_button.label.set_font('serif')
        self.info_button.label.set_fontsize(14)
        self.info_button.on_clicked(self.print_info)

        # Lasso Button

        self.Palette.remove_border(self.axlassobutton)

        self.lasso_button = Button(
            ax=self.axlassobutton,
            label = "Lasso",
            color = self.Palette.slider_colour,
            hovercolor = self.Palette.blue
        )

        self.lasso_button.label.set_color(self.Palette.off_white)
        self.lasso_button.label.set_font('serif')
        self.lasso_button.label.set_fontsize(14)
        self.lasso_button.on_clicked(self.lasso_select)

        # Auto cluster button
        self.Palette.remove_border(self.axclusterbutton)

        self.cluster_button = Button(
            ax=self.axclusterbutton,
            label = "Auto cluster",
            color = self.Palette.slider_colour,
            hovercolor = self.Palette.blue
        )

        self.cluster_button.label.set_color(self.Palette.off_white)
        self.cluster_button.label.set_font('serif')
        self.cluster_button.label.set_fontsize(14)
        self.cluster_button.on_clicked(self.auto_cluster)

        # Decorate clusters axes

        self.axclusters.xaxis.set_tick_params(labelbottom=False)
        self.axclusters.yaxis.set_tick_params(labelleft=False)
        self.axclusters.set_xticks([])
        self.axclusters.set_yticks([])

        self.axclusters.set_title("Clusters", fontdict=self.Palette.subtitle_font)
        self.Palette.remove_border(self.axclusters)


        self.fig.canvas.mpl_connect('pick_event', self.pick_axes)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
