import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, chi2
from calculator import Calculator
from matplotlib.widgets import Button, Slider, CheckButtons, LassoSelector, RadioButtons
from matplotlib.patches import BoxStyle
from matplotlib.path import Path
import warnings
warnings.filterwarnings("error")


base_layout = {
    "ax" : [0.35, 0.1, 0.6, 0.8],
    "axslider" : [0.24, 0.1, 0.02, 0.8],
    "axcheckbox" : [0.1, 0.65, 0.1, 0.25],
    "axrandbutton" : [0.1, 0.58, 0.1, 0.05],
    "axlassobutton" : [0.1, 0.51, 0.1, 0.05],
    "axclusterbutton" : [0.1, 0.44, 0.1, 0.05],
    "axclusters" : [0.1, 0.1, 0.1, 0.3],
    "orientation" : "vertical"
}


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

        ellipse_colour = blue
        ellipse_colours = None
        bg_colour = green
        slider_colour = dark_blue
        slider_bg = np.array([0.969, 0.945, 0.929, 0.5])
        graph_bg = off_white
        points_colour = grey
        redundant_points_colour = graph_bg * 0.9
        axes_colour = black

        cluster_colours = np.outer(np.array([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]), dark_blue)

        title_font = {"color": off_white, "family": "serif"}
        subtitle_font = {"color": off_white, "family": "serif", "size": 14}

        @classmethod
        def __init__(cls) -> None:
            cls.ellipse_colours = np.outer(np.sqrt(np.sqrt(np.reciprocal(np.array(range(1, len(InteractiveGraph.m_dists_using)+1)).astype(float)))), cls.ellipse_colour)
        
        @classmethod
        def remove_border(cls, ax: plt.Axes) -> None:
            ax.spines['bottom'].set_color(cls.bg_colour)
            ax.spines['top'].set_color(cls.bg_colour)
            ax.spines['right'].set_color(cls.bg_colour)
            ax.spines['left'].set_color(cls.bg_colour)

    # constant
    MAX_SD: float = None
    PRECISION: int = None
    PREPLOTS: list[np.ndarray] = None
    CALC: Calculator = None
    CONFS: dict = None
    LIMIT: float = None

    # variable
    curr_proj: np.ndarray = None
    points_ind: int | np.ndarray = 0
    curr_collection = None
    dragged = None
    clusters = None
    m_dists_using: dict = {"1σ": False, "2σ": False, "3σ": True, "4σ": False, "5σ": False, "6σ": False}
    attr_labels = []

    # widgets
    fig = None
    ax: plt.Axes = None
    axslider: plt.Axes = None
    cutoff_slider: Slider = None
    axcheckbox: plt.Axes = None
    m_checkbox: CheckButtons = None
    axrandbutton: plt.Axes = None
    rand_button: Button =  None


    def __init__(self, layout, preplots: int = 20) -> None:
        self.Palette()
 
        readfile = input("Enter path to data csv file: ")
        if readfile == "1":
            readfile = "samples/p5pexample/np.csv"
            cov = np.loadtxt('samples/p5pexample/cov_mat.csv', dtype=float, delimiter=",")
            mean = np.loadtxt('samples/p5pexample/centre_of_ellipses.csv', dtype=float, delimiter=",")
        else:
            manual_cov = input("Would you like to specify a covariance matrix? Y/N: ")
            cov = mean = None
            if manual_cov == 'Y':
                cov_file = input("Enter path to covariance matrix file: ")
                cov = np.loadtxt(cov_file, dtype=float, delimiter=",")
                manual_mean = input("Would you like to specify an ellipsoid mean? Y/N: ")
                if manual_mean == "Y":
                    mean_file = input("Enter path to ellipsoid mean file: ")
                    mean = np.loadtxt(mean_file, dtype=float, delimiter=",")
        self.CALC = Calculator(readfile, cov=cov, cov_mean = mean)

        self.PRECISION: int = preplots if type(preplots) is int else len(preplots)
        self.CONFS: dict = {sdstr : np.sqrt(chi2.ppf((2*norm.cdf(float(sdstr[:-1])) - 1), self.CALC.get_dim())) for sdstr in self.m_dists_using.keys()}
        self.M_DISTS: list[int] = self.CONFS.values()
        self.MAX_SD = self.CALC.get_max_sds()*0.99
        self.LIMIT = self.CALC.get_max_norm()

        self.point_colours = np.vstack([self.Palette.points_colour]*len(self.CALC))
        self.layout = layout
        
        if type(preplots) is int:
            P = None
            for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
                sd = i*self.MAX_SD/self.PRECISION
                u, v = self.CALC.optimise_plane(sd=sd, from_plane=P)
                P = np.vstack([u, v])
                self.PREPLOTS = P[np.newaxis, :, :] if i==0 else np.vstack([self.PREPLOTS, P[np.newaxis, :, :]])
        else:
            self.PREPLOTS = preplots



        self.build_widgets()
        self.update(self.PREPLOTS[self.points_ind])

    def update(self, proj: np.ndarray | None = None):
        if proj is not None:
            self.curr_proj = proj

        self.ax.cla()

        # points
        proj_points = self.curr_proj @ self.CALC.get_data()
        self.curr_collection = self.ax.scatter(proj_points[0], proj_points[1], marker = ".")
        self.curr_collection.set_facecolor(self.point_colours)
        
        # ellipses
        ellipses = self.CALC.get_proj_ellipses(self.curr_proj, [self.CONFS[m] for m in self.CONFS if self.m_dists_using[m]])
        for i, ellipse in enumerate(ellipses):
            self.ax.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        # axes
        i = 0
        self.attr_labels.clear()
        for proj_axis in self.curr_proj.T:
            self.ax.plot([0, proj_axis[0]*self.LIMIT*0.5], [0, proj_axis[1]*self.LIMIT*0.5], c = self.Palette.axes_colour, linewidth = 1)
            new_label = self.ax.text(proj_axis[0]*self.LIMIT*0.5, proj_axis[1]*self.LIMIT*0.5, self.CALC.get_attrs()[i], picker=True)
            #new_label.set_picker(True)
            new_label.set_bbox(dict(facecolor=self.Palette.graph_bg, alpha=0.7, linewidth=0, boxstyle=BoxStyle.Round(pad=0.05)))
            self.attr_labels.append(new_label)
            i += 1
        
        self.ax.set_xbound(-self.LIMIT, self.LIMIT)
        self.ax.set_ybound(-self.LIMIT, self.LIMIT)

        self.fig.canvas.draw_idle()

    def pick_axes(self, event):
        if event.mouseevent.dblclick:
            ind = np.where(self.CALC.get_attrs() == event.artist.get_text())[0][0]
            u, v = self.curr_proj[0], self.curr_proj[1]
            u[ind] = v[ind] = 0
            try:
                u, v = self.CALC.orthonormalise(u, v)
            except RuntimeWarning:
                pass
            else:
                self.curr_proj = np.vstack([u, v])
                self.update()
        else:
            self.dragged = np.where(self.CALC.get_attrs() == event.artist.get_text())[0][0]
            self.dim2 = False
            self.drag_id = self.fig.canvas.mpl_connect("motion_notify_event", self.drag_axes)
            self.release_id = self.fig.canvas.mpl_connect('button_release_event', self.stop_dragging)
            
    def drag_axes(self, event):
        u, v = self.curr_proj[0], self.curr_proj[1]

        pos = np.array([event.xdata*2/self.LIMIT, event.ydata*2/self.LIMIT])
        precision = 0.00001
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

        a, b = au*np.sqrt(1-A**2), bu*np.sqrt(1-B**2)
        u = np.insert(a, self.dragged, A)
        v = np.insert(b, self.dragged, B)
        
        #u[self.dragged] = event.xdata*2/self.LIMIT
        #v[self.dragged] = event.ydata*2/self.LIMIT
        #u, v = self.CALC.orthonormalise(u, v)
        
        self.update(np.vstack([u, v]))

    def stop_dragging(self, event):
        self.dragged = None
        self.fig.canvas.mpl_disconnect(self.drag_id)
        self.fig.canvas.mpl_disconnect(self.release_id)

    def show_clusters(self):
        pass

    def change_cutoff(self, val = None):
        proj_ind = self.cutoff_slider.val*self.PRECISION/self.MAX_SD
        if int(proj_ind) >= self.PRECISION - 1:
            proj = self.PREPLOTS[self.PRECISION-1]
        else:
            proj_low = self.PREPLOTS[int(proj_ind)]
            proj_high = self.PREPLOTS[int(proj_ind) + 1]
            frac = proj_ind - int(proj_ind)
            proj = proj_low + frac*(proj_high - proj_low)
        
        self.points_ind = self.CALC.partition_data(self.cutoff_slider.val)
        self.point_colours[self.points_ind:] = self.Palette.points_colour
        self.point_colours[:self.points_ind] = self.Palette.redundant_points_colour
        self.update(proj)

    def show_random(self, event):
        P = self.CALC.get_random_plane()
        self.update(P)

    def lasso_select(self, event):
        self.lasso_button.color = self.Palette.blue
        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.fig.canvas.mpl_connect("key_press_event", self.lasso_accept)

    def on_select(self, verts):
        path = Path(verts)
        self.suggest_ind = np.nonzero(path.contains_points(self.curr_collection.get_offsets()))[0]
        self.point_colours[:] = self.Palette.redundant_points_colour
        self.point_colours[self.suggest_ind] = self.Palette.points_colour
        
        self.update()
        self.ax.set_title("Press enter to accept selection")
        self.fig.canvas.draw_idle()

    def lasso_accept(self, event):
        if event.key == "enter":

            self.lasso_button.color = self.Palette.slider_colour
            self.points_ind = self.suggest_ind
            self.suggest_ind = None
            points = self.CALC.get_data()[:, self.points_ind]

            self.ax.set_title("Loading projection...")
            self.fig.canvas.draw_idle()
            u, v = self.CALC.optimise_plane(points=points, from_plane=self.curr_proj)
            P = np.vstack([u, v])
            self.lasso.disconnect_events()
            self.update(P)

    def show_ellipse(self, label):
        self.m_dists_using[label] = not self.m_dists_using[label]
        self.update()

    def build_widgets(self):
        # figure and axes
        self.fig = plt.figure(facecolor=self.Palette.bg_colour)
        self.ax = self.fig.add_axes(self.layout["ax"], facecolor = self.Palette.graph_bg)
        self.axslider = self.fig.add_axes(self.layout["axslider"], facecolor=self.Palette.slider_bg)
        self.axcheckbox = self.fig.add_axes(self.layout["axcheckbox"], facecolor=self.Palette.slider_bg)
        self.axrandbutton = self.fig.add_axes(self.layout["axrandbutton"])
        self.axlassobutton = self.fig.add_axes(self.layout["axlassobutton"])
        self.axclusterbutton = self.fig.add_axes(self.layout["axclusterbutton"])
        self.axclusters = self.fig.add_axes(self.layout["axclusters"], facecolor=self.Palette.slider_bg)

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
            orientation="vertical",
            color=self.Palette.slider_colour,
            track_color=self.Palette.slider_bg,
            closedmax=False,
            initcolor=None
        )
        self.cutoff_slider.on_changed(self.change_cutoff)

        # Checkboxes

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


        # Button

        self.Palette.remove_border(self.axrandbutton)

        self.rand_button = Button(
            ax=self.axrandbutton,
            label = "Random",
            color = self.Palette.slider_colour,
            hovercolor = self.Palette.blue
        )

        self.rand_button.label.set_color(self.Palette.off_white)
        self.rand_button.label.set_font('serif')
        self.rand_button.label.set_fontsize(14)
        self.rand_button.on_clicked(self.show_random)

        # Button

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


        self.Palette.remove_border(self.axclusterbutton)

        self.cluster_button = Button(
            ax=self.axclusterbutton,
            label = "Cluster",
            color = self.Palette.slider_colour,
            hovercolor = self.Palette.blue
        )

        self.cluster_button.label.set_color(self.Palette.off_white)
        self.cluster_button.label.set_font('serif')
        self.cluster_button.label.set_fontsize(14)
        self.cluster_button.on_clicked(self.show_clusters)


        self.axclusters.set_title("Clusters", fontdict=self.Palette.subtitle_font)
        self.Palette.remove_border(self.axclusters)

        self.fig.canvas.mpl_connect('pick_event', self.pick_axes)
        self.fig.suptitle('Projected data', fontdict=self.Palette.title_font, fontsize=24)

#class InteractiveFunction(InteractiveGraph):


# Save to binary file
if __name__=='__main__':
    this_graph = InteractiveGraph(base_layout)
    #writefile = input("Enter path to save widget (.pickle extension): ")
    #pl.dump((this_graph.PREPLOTS, this_graph.CALC, this_graph.limits), open(writefile,'wb'))
    plt.show()