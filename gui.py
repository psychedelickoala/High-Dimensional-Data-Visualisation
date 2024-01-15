import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, chi2
from calculator import Calculator
from matplotlib.widgets import Button, Slider, CheckButtons, LassoSelector, RadioButtons
from matplotlib.patches import BoxStyle
from matplotlib.path import Path
import pickle as pl
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

        ellipse_colour = blue
        ellipse_colours = None
        bg_colour = green
        slider_colour = dark_blue
        slider_bg = np.array([0.969, 0.945, 0.929, 0.5])
        graph_bg = off_white
        points_colour = grey
        redundant_points_colour = graph_bg * 0.9
        axes_colour = black

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
    FACTORS: list = [0.985, None, 1.05]
    CONFS: dict = None


    # variable
    curr_proj: np.ndarray = None
    axes_zeroed = None
    points_ind: float | np.ndarray = 0
    curr_collection = None
    colours = None
    weight_index = None
    dragged = None
    m_dists_using = dict = {
        "1σ": False,
        "2σ": False,
        "3σ": True,
        "4σ": False,
        "5σ": False,
        "6σ": False
    }

    # widgets
    fig = None
    ax: plt.Axes = None
    axslider: plt.Axes = None
    cutoff_slider: Slider = None
    axcheckbox: plt.Axes = None
    m_checkbox: CheckButtons = None
    axrandbutton: plt.Axes = None
    rand_button: Button =  None

    def __init__(self, preplots: list[np.ndarray] | int = 200, calc: Calculator = None) -> None:
        self.Palette()
        if calc is None:
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
        else:
            self.CALC = calc
        
        self.PRECISION = preplots if type(preplots) is int else len(preplots)
        self.CONFS: dict = {
            "1σ": np.sqrt(chi2.ppf((2*norm.cdf(1) - 1), self.CALC.get_dim())),
            "2σ": np.sqrt(chi2.ppf((2*norm.cdf(2) - 1), self.CALC.get_dim())),
            "3σ": np.sqrt(chi2.ppf((2*norm.cdf(3) - 1), self.CALC.get_dim())),
            "4σ": np.sqrt(chi2.ppf((2*norm.cdf(4) - 1), self.CALC.get_dim())),
            "5σ": np.sqrt(chi2.ppf((2*norm.cdf(5) - 1), self.CALC.get_dim())),
            "6σ": np.sqrt(chi2.ppf((2*norm.cdf(6) - 1), self.CALC.get_dim()))
        }
        self.M_DISTS: list[int] = self.CONFS.values()
        self.MAX_SD = self.CALC.get_max_sds()*0.99
        self.point_colours = np.vstack([self.Palette.points_colour]*len(self.CALC))
        self.axes_zeroed = [False]*self.CALC.get_dim()
        self.weight_index = 1
        self.points_ind = 0
        
        # figure and axes
        self.fig = plt.figure(facecolor=self.Palette.bg_colour)
        self.ax = self.fig.add_axes([0.35, 0.1, 0.6, 0.8], facecolor = self.Palette.graph_bg)
        self.axslider = self.fig.add_axes([0.24, 0.1, 0.02, 0.8], facecolor=self.Palette.slider_bg)
        self.axcheckbox = self.fig.add_axes([0.1, 0.55, 0.1, 0.35], facecolor=self.Palette.slider_bg)
        self.axrandbutton = self.fig.add_axes([0.1, 0.45, 0.1, 0.05])
        self.axlassobutton = self.fig.add_axes([0.1, 0.35, 0.1, 0.05])
        self.axradio = self.fig.add_axes([0.1, 0.1, 0.1, 0.15], facecolor = self.Palette.slider_bg)

        
        if type(preplots) is int:

            P_in = P_mid = P_out = None
            #self.PREPLOTS = np.array([])
            for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
                sd = i*self.MAX_SD/self.PRECISION

                #u, v = self.CALC.optimise_plane(sd=sd, factor = self.FACTORS[0], from_plane=P_in)
                #P_in = np.vstack([u, v])
                u, v = self.CALC.optimise_plane(sd=sd, factor = self.FACTORS[1], from_plane=P_mid)
                P_mid = np.vstack([u, v])
                #u, v = self.CALC.optimise_plane(sd=sd, factor = self.FACTORS[2], from_plane=P_out)
                #P_out = np.vstack([u, v])
                row = np.stack([P_mid])[np.newaxis]
                if i == 0:
                    self.PREPLOTS = row
                else:
                    self.PREPLOTS = np.vstack([self.PREPLOTS, row])

        else:
            self.PREPLOTS = preplots

        self.limit = self.CALC.get_max_norm()
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


        self.Palette.remove_border(self.axradio)
        self.axradio.set_title("Weighting", fontdict=self.Palette.subtitle_font)

        self.factor_radio = RadioButtons(
            ax = self.axradio,
            labels=["Inner", "None", "Outer"],
            label_props = {"color": [self.Palette.green, self.Palette.blue, self.Palette.dark_blue], "family":["serif"]*3, "size":[14]*3},
            active = 1,
            radio_props = {"facecolor": [self.Palette.green, self.Palette.blue, self.Palette.dark_blue],
                           "edgecolor": [0.8*self.Palette.green, 0.8*self.Palette.blue, 0.8*self.Palette.dark_blue]}
        )
        self.factor_radio.on_clicked(self.change_factor)

        self.fig.canvas.mpl_connect('pick_event', self.pick_axes)
        self.update(self.PREPLOTS[self.points_ind, 0])

        self.fig.suptitle('Projected data', fontdict=self.Palette.title_font, fontsize=24)

    
    def update(self, proj: np.ndarray | None = None):
        if proj is not None:
            self.curr_proj = proj

        self.ax.cla()
        proj_points = self.curr_proj @ self.CALC.get_data()
        self.curr_collection = self.ax.scatter(proj_points[0], proj_points[1], marker = ".")
        self.curr_collection.set_facecolor(self.point_colours)
        
        #self.ax.scatter(self.curr_proj.black_points[0], self.curr_proj.black_points[1], c = [self.Palette.points_colour], marker = ".")
        ellipses = self.CALC.get_proj_ellipses(self.curr_proj, [self.CONFS[m] for m in self.CONFS if self.m_dists_using[m]])
        for i, ellipse in enumerate(ellipses):
            self.ax.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        i = 0
        self.attr_labels = []
        for proj_axis in self.curr_proj.T:
            self.ax.plot([0, proj_axis[0]*self.limit*0.5], [0, proj_axis[1]*self.limit*0.5], c = self.Palette.axes_colour, linewidth = 1)
            new_label = self.ax.text(proj_axis[0]*self.limit*0.5, proj_axis[1]*self.limit*0.5, self.CALC.get_attrs()[i])
            new_label.set_picker(True)
            new_label.set_bbox(dict(facecolor=self.Palette.graph_bg, alpha=0.7, linewidth=0, boxstyle=BoxStyle.Round(pad=0.05)))
            self.attr_labels.append(new_label)
            i += 1
        
        self.ax.set_xbound(-self.limit, self.limit)
        self.ax.set_ybound(-self.limit, self.limit)

        self.fig.canvas.draw_idle()

    def pick_axes(self, event):
        if event.mouseevent.dblclick:
            ind = np.where(self.CALC.get_attrs() == event.artist.get_text())[0][0]
            u, v = self.curr_proj[0], self.curr_proj[1]
            u[ind] *= 0
            v[ind] *= 0
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
        nonzeros = 0
        for row in self.curr_proj.T:
            if not np.allclose(row, np.zeros(2)):
                nonzeros += 1

        dim2 = (nonzeros <= 2)
        pos = np.array([event.xdata*2/self.limit, event.ydata*2/self.limit])
        precision = 0.00001
        a = np.delete(u, self.dragged)
        b = np.delete(v, self.dragged)
        A = pos[0] if (np.linalg.norm(pos) < (1-precision) and not dim2) else (1 - precision)*(pos[0])/np.linalg.norm(pos)
        B = pos[1] if (np.linalg.norm(pos) < (1-precision) and not dim2) else (1 - precision)*(pos[1])/np.linalg.norm(pos)

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


        '''
        a, b = au*np.sqrt(1-A**2), bu*np.sqrt(1-B**2)
        u = np.insert(a, self.dragged, A)
        v = np.insert(b, self.dragged, B)
        '''


        u[self.dragged] = event.xdata*2/self.limit
        v[self.dragged] = event.ydata*2/self.limit
        u, v = self.CALC.orthonormalise(u, v)
        
        self.update(np.vstack([u, v]))

    def stop_dragging(self, event):
        self.dragged = None
        self.fig.canvas.mpl_disconnect(self.drag_id)
        self.fig.canvas.mpl_disconnect(self.release_id)


    def change_factor(self, label):
        radio_labels = {"Inner": 0, "None": 1, "Outer": 2}
        self.weight_index = radio_labels[label]

        points = self.CALC.get_data()[:, self.points_ind]
        u, v = self.CALC.optimise_plane(points=points, from_plane=self.curr_proj, factor=self.FACTORS[self.weight_index])
        P = np.vstack([u, v])
        self.update(P)

    def change_cutoff(self, val = None):
        proj = self.PREPLOTS[int(self.cutoff_slider.val*self.PRECISION/self.MAX_SD), 0]
        self.factor_radio.set_active(1)
        self.points_ind = self.CALC.partition_data(self.cutoff_slider.val)
        self.point_colours[self.points_ind:] = self.Palette.points_colour
        self.point_colours[:self.points_ind] = self.Palette.redundant_points_colour
        self.update(proj)

    def show_random(self, event):
        P = self.CALC.get_random_plane()
        self.update(P)

    def lasso_select(self, event):
        self.lasso_button.color = self.Palette.blue
        self.selector = SelectFromCollection(self.ax, self.curr_collection, self)
        self.fig.canvas.mpl_connect("key_press_event", self.lasso_accept)

    def lasso_accept(self, event):
        if event.key == "enter":
            #print("Selected points:")
            self.lasso_button.color = self.Palette.slider_colour
            self.points_ind = self.selector.ind
            points = self.CALC.get_data()[:, self.points_ind]
            #print(selector.xys[selector.ind])
            self.ax.set_title("Loading projection...")
            u, v = self.CALC.optimise_plane(points=points, from_plane=self.curr_proj, factor=self.FACTORS[self.weight_index])
            P = np.vstack([u, v])
            self.update(P)
            
            self.selector.disconnect()
            self.fig.canvas.draw_idle()

    def show_ellipse(self, label):
        self.m_dists_using[label] = not self.m_dists_using[label]
        self.update()


# Lasso
class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, graph_obj):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.graph = graph_obj

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self.fc = self.graph.point_colours

        # Ensure that we have separate colors for each object
        #self.fc = collection.get_facecolors()
        #if len(self.fc) == 0:
        #    raise ValueError('Collection must have a facecolor')
        #elif len(self.fc) == 1:
        #    self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:] = self.graph.Palette.redundant_points_colour
        self.fc[self.ind] = self.graph.Palette.points_colour
        self.graph.point_colours = self.fc
        self.graph.update()
        self.graph.ax.set_title("Press enter to accept selection")
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        #cutoff_ind = self.graph.CALC.partition_data(InteractiveGraph.cutoff_slider.val)
        #self.fc[:cutoff_ind] = self.graph.Palette.redundant_points_colour
        #self.fc[cutoff_ind:] = self.graph.Palette.points_colour
        #self.collection.set_facecolors(self.fc)
        #self.graph.change_cutoff()
        self.canvas.draw_idle()





# Save to binary file
if __name__=='__main__':
    this_graph = InteractiveGraph(200)
    #writefile = input("Enter path to save widget (.pickle extension): ")
    #pl.dump((this_graph.PREPLOTS, this_graph.CALC, this_graph.limits), open(writefile,'wb'))
    plt.show()