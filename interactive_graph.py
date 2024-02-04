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

    # Colours
    class Palette:
        light_green = np.array([0.898, 0.988, 0.761])
        green = np.array([0.616, 0.878, 0.678])
        blue = np.array([0.271, 0.678, 0.659])
        dark_blue = np.array([0.329, 0.475, 0.502])
        grey = np.array([0.349, 0.31, 0.31])
        off_white = np.array([0.969, 0.945, 0.929])
        black = np.array([0, 0, 0])

        suggest_colour = np.array([0.329, 0.475, 0.502, 1])
        ellipse_colour = blue
        ellipse_colours = None
        bg_colour = green
        slider_colour = dark_blue
        slider_bg = np.array([0.969, 0.945, 0.929, 0.5])
        graph_bg = off_white
        points_colour = grey
        redundant_points_colour = graph_bg * 0.9
        axes_colour = black

        #cluster_colours = np.outer(np.array([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]), dark_blue)
        cluster_colours = np.array([[80, 80, 80, 255],
            [255, 84, 0, 255], [255, 142, 0, 255], [255, 210, 0, 255],
            [129, 230, 80, 255], [0, 210, 103, 255], [0, 192, 255, 255],
            [139, 72, 254, 255], [202, 65, 252, 255], [255, 70, 251, 255]
        ])/255

        #cluster_colours_light = (np.array([1, 1, 1]) - cluster_colours)*0.5 + cluster_colours

        title_font = {"color": off_white, "family": "serif"}
        subtitle_font = {"color": off_white, "family": "serif", "size": 14}

        @classmethod
        def __init__(cls, num_points, num_ellipses) -> None:
            cls.ellipse_colours = np.outer(np.sqrt(np.sqrt(np.reciprocal(np.array(range(1, num_ellipses+1)).astype(float)))), cls.ellipse_colour)
            alphas = (np.ones((10, 1)))/np.sqrt(num_points)
            cls.cluster_colours_light = np.hstack([cls.cluster_colours[:, :3], alphas])
        
        @classmethod
        def remove_border(cls, ax: plt.Axes) -> None:
            ax.spines['bottom'].set_color(cls.bg_colour)
            ax.spines['top'].set_color(cls.bg_colour)
            ax.spines['right'].set_color(cls.bg_colour)
            ax.spines['left'].set_color(cls.bg_colour)

    # constant
    MAX_SD: float = None
    PRECISION: int = 20
    PREPLOTS: list[np.ndarray] = None
    CALC: Calculator = None
    CONFS: dict = None
    LIMIT: float = None
    LAYOUT: dict = {
        "ax" : [0.35, 0.1, 0.6, 0.8],
        "axdep" : None,
        "axslider" : [0.24, 0.1, 0.02, 0.8],
        "axcheckbox" : [0.1, 0.65, 0.1, 0.25],
        "axrandbutton" : [0.1, 0.58, 0.1, 0.05],
        "axlassobutton" : [0.1, 0.51, 0.1, 0.05],
        "axclusterbutton" : [0.1, 0.44, 0.1, 0.05],
        "axclusters" : [0.1, 0.1, 0.1, 0.3],
        "orientation" : "vertical"
    }
    LAYOUT["cluster"] = [None,
        [(0.00, 0.67), 0.33, 0.33], [(0.33, 0.67), 0.34, 0.33], [(0.67, 0.67), 0.33, 0.33],
        [(0.00, 0.33), 0.33, 0.34], [(0.33, 0.33), 0.34, 0.34], [(0.67, 0.33), 0.33, 0.34],
        [(0.00, 0.00), 0.33, 0.33], [(0.33, 0.00), 0.34, 0.33], [(0.67, 0.00), 0.33, 0.33]
    ]

    # variable
    curr_proj: np.ndarray = None
    points_ind: int | np.ndarray = 0
    curr_collection = None
    dragged = None
    clusters = None
    clusters_in_use = [0]
    m_dists_using: dict = {"1σ": False, "2σ": True, "3σ": False, "5σ": False, "8σ": False, "13σ": False}
    attr_labels = []
    suggest_ind = None
    lassoing = False

    # widgets
    fig = None
    ax: plt.Axes = None
    axdep = None
    axslider: plt.Axes = None
    cutoff_slider: Slider = None
    axcheckbox: plt.Axes = None
    m_checkbox: CheckButtons = None
    axrandbutton: plt.Axes = None
    rand_button: Button =  None


    def __init__(self, data, cov_data = None, mean_data = None, update = True) -> None:
 
        self.CALC = Calculator(data=data, cov=cov_data, cov_mean=mean_data)
        self.Palette(len(self.CALC), len(self.m_dists_using))
        #self.PRECISION: int = preplots if type(preplots) is int else len(preplots)
        self.CONFS: dict = {sdstr : np.sqrt(chi2.ppf((2*norm.cdf(float(sdstr[:-1])) - 1), self.CALC.get_dim())) for sdstr in self.m_dists_using.keys()}
        self.M_DISTS: list[int] = self.CONFS.values()
        self.MAX_SD = self.CALC.get_max_sds()*0.99
        self.LIMIT = self.CALC.get_max_norm()

        self.point_colours = np.vstack([self.Palette.points_colour]*len(self.CALC))
        self.clusters = np.array([0]*len(self.CALC))
        self.num_clusters = 0
        
        P = None
        for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
            sd = i*self.MAX_SD/self.PRECISION
            u, v = self.CALC.optimise_plane(sd=sd, from_plane=P)
            P = np.vstack([u, v])
            self.PREPLOTS = P[np.newaxis, :, :] if i==0 else np.vstack([self.PREPLOTS, P[np.newaxis, :, :]])

        self.build_widgets()
        if update:
            self.update(self.PREPLOTS[self.points_ind])

    def update(self, proj: np.ndarray | None = None, ax = None, calc = None, labels = None):
        if proj is not None:
            self.curr_proj = proj
        if ax is None:
            ax = self.ax
        if calc is None:
            calc = self.CALC
        if labels is None:
            labels = self.attr_labels

        ax.cla()

        # points
        point_colours = self.get_point_colours()
        proj_mean = self.curr_proj @ calc.get_cov_mean()
        proj_points = self.curr_proj @ calc.get_data() - proj_mean
        self.curr_collection = ax.scatter(proj_points[0], proj_points[1], marker = ".")
        self.curr_collection.set_facecolor(point_colours)
        
        # ellipses
        ellipses = calc.get_proj_ellipses(self.curr_proj, [self.CONFS[m] for m in self.CONFS if self.m_dists_using[m]])
        for i, ellipse in enumerate(ellipses):
            ax.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        # axes
        i = 0
        labels.clear()
        for proj_axis in self.curr_proj.T:
            ax.plot([0, proj_axis[0]*calc.get_max_norm()*0.5], [0, proj_axis[1]*calc.get_max_norm()*0.5], c = self.Palette.axes_colour, linewidth = 1)
            new_label = ax.text(proj_axis[0]*calc.get_max_norm()*0.5, proj_axis[1]*calc.get_max_norm()*0.5, calc.get_attrs()[i], picker=True)
            #new_label.set_picker(True)
            new_label.set_bbox(dict(facecolor=self.Palette.graph_bg, alpha=0.7, linewidth=0, boxstyle=BoxStyle.Round(pad=0.05)))
            labels.append(new_label)
            i += 1
        

        ax.set_xbound(-calc.get_max_norm(), calc.get_max_norm())
        ax.set_ybound(-calc.get_max_norm(), calc.get_max_norm())
        
        # Hide X and Y axes label marks
        self.axclusters.xaxis.set_tick_params(labelbottom=False)
        self.axclusters.yaxis.set_tick_params(labelleft=False)

        self.fig.canvas.draw_idle()

    def get_point_colours(self) -> np.ndarray:
        point_colours = self.Palette.cluster_colours_light[self.clusters]
        
        #dark_colours = self.Palette.cluster_colours[self.clusters]
        #clusters = self.clusters[self.points_ind] if type(self.points_ind) is np.ndarray else self.clusters[self.points_ind:]
        #print(self.Palette.cluster_colours[clusters])
        if type(self.points_ind) is np.ndarray:
            point_colours[self.points_ind] = self.Palette.cluster_colours[self.clusters[self.points_ind]]
        else:
            point_colours[self.points_ind:] = self.Palette.cluster_colours[self.clusters[self.points_ind:]]
        if self.suggest_ind is not None:
            point_colours[self.suggest_ind] = self.Palette.suggest_colour
        return point_colours

    def pick_axes(self, event, calc = None, proj = None, update = True):
        if calc is None:
            calc = self.CALC
        if proj is None:
            proj = self.curr_proj

        if isinstance(event.artist, Rectangle):
            this_cluster = int(event.artist.get_gid())
            if event.mouseevent.dblclick:
                # delete cluster
                self.clusters_in_use.remove(this_cluster)
                self.clusters[np.where(self.clusters == this_cluster)[0]] = 0
                self.num_clusters -= 1
                event.artist.remove()
                self.update()
            elif event.mouseevent.button is MouseButton.RIGHT:
                self.points_ind = np.where(self.clusters == this_cluster)[0]
                points = calc.get_data()[:, self.points_ind]
                u, v = calc.optimise_plane(points=points, from_plane=proj)
                new_proj = np.vstack([u, v])
                if update:
                    self.update(new_proj)
                else:
                    return new_proj
        
        elif event.mouseevent.dblclick:
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
        else:
            self.dragged = np.where(calc.get_attrs() == event.artist.get_text())[0][0]
            self.dim2 = False
            self.drag_id = self.fig.canvas.mpl_connect("motion_notify_event", self.drag_axes)
            self.release_id = self.fig.canvas.mpl_connect('button_release_event', self.stop_dragging)
            
    def drag_axes(self, event, proj = None, calc = None, update = True):
        if proj is None:
            proj = self.curr_proj
        if calc is None:
            calc = self.CALC
        u, v = proj[0], proj[1]
        try:
            pos = np.array([event.xdata*2/calc.get_max_norm(), event.ydata*2/calc.get_max_norm()])
        except:
            self.stop_dragging()
            return
        precision = 0.0000
        a = np.delete(u, self.dragged)
        b = np.delete(v, self.dragged)
        A = pos[0] if np.linalg.norm(pos) < (1-precision) else (1 - precision)*(pos[0])/np.linalg.norm(pos)
        B = pos[1] if np.linalg.norm(pos) < (1-precision) else (1 - precision)*(pos[1])/np.linalg.norm(pos)

        au = a/np.linalg.norm(a)
        bu = b/np.linalg.norm(b)
        m = (au + bu)
        try:
            m /= np.linalg.norm(m)
            c2 = -A*B/np.sqrt((1 - A**2)*(1 - B**2))
            c, s = np.sqrt((1 + c2)/2), np.sqrt((1 - c2)/2)

            ap = (au - np.dot(au, m)*m)
            ap /= np.linalg.norm(ap)
            bp = (bu - np.dot(bu, m)*m)
            bp /= np.linalg.norm(bp)

            au, bu = c*m + s*ap, c*m + s*bp
        
        except RuntimeWarning: # all other vectors in a line
            u[self.dragged] = pos[0]
            v[self.dragged] = pos[1]
            u, v = self.CALC.orthonormalise(u, v)
            '''
            new_line_proj = np.array([[B**2, -A*B], [-A*B, A**2]])
            to_proj = np.vstack([a, b])
            new_vecs = new_line_proj @ to_proj
            #print(new_vecs[0], new_vecs[1])
            try:
                au = new_vecs[0]/np.linalg.norm(new_vecs[0])
                bu = new_vecs[1]/np.linalg.norm(new_vecs[1])
            except RuntimeWarning:
                if np.abs(A) == 1:
                    au = new_vecs[0]
                    bu = new_vecs[1]/np.linalg.norm(new_vecs[1])
                elif np.abs(B) == 1:
                    au = new_vecs[0]/np.linalg.norm(new_vecs[0])
                    bu = new_vecs[1]
            '''
        else:

            a, b = au*np.sqrt(1-A**2), bu*np.sqrt(1-B**2)
            u = np.insert(a, self.dragged, A)
            v = np.insert(b, self.dragged, B)
        
        #u[self.dragged] = event.xdata*2/self.LIMIT
        #v[self.dragged] = event.ydata*2/self.LIMIT
        #u, v = self.CALC.orthonormalise(u, v)
        if update:
            self.update(np.vstack([u, v]))
        else:
            return np.vstack([u, v])

    def stop_dragging(self, event = None):
        self.dragged = None
        self.fig.canvas.mpl_disconnect(self.drag_id)
        self.fig.canvas.mpl_disconnect(self.release_id)

    def change_cutoff(self, val = None, pres = None, update = True):
        if pres is None:
            pres = self.PREPLOTS
        proj_ind = self.cutoff_slider.val*self.PRECISION/self.MAX_SD
        if int(proj_ind) >= self.PRECISION - 1:
            proj = pres[self.PRECISION-1]
        else:
            proj_low = pres[int(proj_ind)]
            proj_high = pres[int(proj_ind) + 1]
            frac = proj_ind - int(proj_ind)
            proj = proj_low + frac*(proj_high - proj_low)
        
        self.points_ind = self.CALC.partition_data(self.cutoff_slider.val)
        self.point_colours[self.points_ind:] = self.Palette.points_colour
        self.point_colours[:self.points_ind] = self.Palette.redundant_points_colour
        if update:
            self.update(proj)
        else:
            return proj

    def show_random(self, event = None):
        P = self.CALC.get_random_plane()
        self.update(P)

    def lasso_select(self, event):
        self.lasso_button.color = self.Palette.blue
        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.lassoing = True

    def on_select(self, verts):
        path = Path(verts)
        self.suggest_ind = np.nonzero(path.contains_points(self.curr_collection.get_offsets()))[0]
        self.point_colours[:] = self.Palette.redundant_points_colour
        self.point_colours[self.suggest_ind] = self.Palette.points_colour
        
        self.update()
        self.ax.set_title("Press enter to change projection, Y to add cluster")
        self.fig.canvas.draw_idle()

    def move_by_select(self, calc = None, proj = None, update = True):
        if calc is None:
            calc = self.CALC
        if proj is None:
            proj = self.curr_proj
        if self.suggest_ind is not None:
            self.points_ind = self.suggest_ind
            self.suggest_ind = None
        points = calc.get_data()[:, self.points_ind]

        self.ax.set_title("Loading projection...")
        self.fig.canvas.draw_idle()
        u, v = calc.optimise_plane(points=points, from_plane=proj)
        
        if update:
            self.curr_proj = np.vstack([u, v])
            self.lasso.disconnect_events()
        else:
            return np.vstack([u, v])

        #self.update(P)

    def add_cluster(self):
        if self.num_clusters == 9:
            self.lasso.disconnect_events()
            self.suggest_ind = None
            return
        self.num_clusters += 1
        this_cluster = [i for i in range(10) if i not in self.clusters_in_use][0]
        self.clusters_in_use.append(this_cluster)
        self.clusters[self.suggest_ind] = this_cluster
        self.suggest_ind = None
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
        #self.update()

    def clear_clusters(self):
        self.clusters = np.array([0]*len(self.CALC))
        self.num_clusters = 0
        self.clusters_in_use = [0]
        self.axclusters.cla()
        # Hide X and Y axes tick marks
        self.axclusters.set_xticks([])
        self.axclusters.set_yticks([])
        self.update()

    def auto_cluster(self, event = None):
        self.clear_clusters()
        self.ax.set_title("Clustering...")
        self.fig.canvas.draw()
        num_clusters, new_clusters = self.CALC.get_clusters(self.points_ind)
        self.num_clusters = num_clusters
        #for i in range(len(new_clusters)):
        #    new_clusters[i] = 0 if new_clusters[i] > 9 else new_clusters[i]
        if type(self.points_ind) is np.ndarray:
            self.clusters[self.points_ind] = new_clusters
        else:
            self.clusters[self.points_ind:] = new_clusters
        for i in range(1, min(num_clusters + 1, 10)):
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
        self.ax.set_title('')
        self.update()

    def print_info(self, event = None):
        print("\n~~ CONTROLS ~~")
        print("Key presses")
        print("R: Show random projection")
        print("C: Clear all clusters")
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
        if self.lassoing:
            self.lassoing = False
            self.lasso_button.color = self.Palette.slider_colour
            if event.key == "enter":
                self.move_by_select()
            elif event.key == "y":
                self.add_cluster()
            else:
                self.suggest_ind = None
                self.lasso.disconnect_events()
            self.update()

        elif event.key == "r":
            self.show_random(event)

        elif event.key == "c":
            self.clear_clusters()

        elif event.key == "m":
            print("\n~~ SELECTED PROJECTION ~~")
            print(np.round(self.curr_proj, 4))

        elif event.key == "n":
            ids = self.CALC.get_sort()[self.points_ind] if type(self.points_ind) is np.ndarray else self.CALC.get_sort()[self.points_ind:]
            print("\n~~ SELECTED POINT IDS ~~")
            print(np.sort(ids) + self.CALC.csv_dist)

    def show_ellipse(self, label):
        self.m_dists_using[label] = not self.m_dists_using[label]
        self.update()

    def build_widgets(self, ellipses = True):
        # figure and axes
        self.fig = plt.figure(facecolor=self.Palette.bg_colour)
        self.ax = self.fig.add_axes(self.LAYOUT["ax"], facecolor = self.Palette.graph_bg)
        self.axslider = self.fig.add_axes(self.LAYOUT["axslider"], facecolor=self.Palette.slider_bg)
        self.axcheckbox = self.fig.add_axes(self.LAYOUT["axcheckbox"], facecolor=self.Palette.slider_bg)
        self.axrandbutton = self.fig.add_axes(self.LAYOUT["axrandbutton"])
        self.axlassobutton = self.fig.add_axes(self.LAYOUT["axlassobutton"])
        self.axclusterbutton = self.fig.add_axes(self.LAYOUT["axclusterbutton"])
        self.axclusters = self.fig.add_axes(self.LAYOUT["axclusters"], facecolor=self.Palette.slider_bg)

        # Hide X and Y axes label marks
        self.axclusters.xaxis.set_tick_params(labelbottom=False)
        self.axclusters.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        self.axclusters.set_xticks([])
        self.axclusters.set_yticks([])

        self.ax.set_aspect('equal', adjustable='box')

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

        self.Palette.remove_border(self.axrandbutton)

        self.rand_button = Button(
            ax=self.axrandbutton,
            label = "Print help",
            color = self.Palette.slider_colour,
            hovercolor = self.Palette.blue
        )

        self.rand_button.label.set_color(self.Palette.off_white)
        self.rand_button.label.set_font('serif')
        self.rand_button.label.set_fontsize(14)
        self.rand_button.on_clicked(self.print_info)

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

        # cluster button
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



        self.axclusters.set_title("Clusters", fontdict=self.Palette.subtitle_font)
        self.Palette.remove_border(self.axclusters)


        self.fig.canvas.mpl_connect('pick_event', self.pick_axes)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        #self.fig.suptitle('Projected data', fontdict=self.Palette.title_font, fontsize=24)

'''
# Save to binary file
if __name__=='__main__':
    #this_graph = InteractiveGraph(data="samples/p5pexample/np.csv", cov_data="samples/p5pexample/cov_mat.csv", mean_data="samples/p5pexample/centre_of_ellipses.csv")
    this_graph=  InteractiveGraph(data="samples/spaced_clusters.csv")
    #ifunc = InteractiveFunction(data="samples/v1.csv", dep_data="samples/two_groups.csv")
    #writefile = input("Enter path to save widget (.pickle extension): ")
    #pl.dump((this_graph.PREPLOTS, this_graph.CALC, this_graph.limits), open(writefile,'wb'))
    plt.show()
'''