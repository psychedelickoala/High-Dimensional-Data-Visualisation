import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from calculator import Calculator
from matplotlib.widgets import Button, Slider, CheckButtons, LassoSelector
from matplotlib.path import Path
import pickle as pl

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
            cls.ellipse_colours = np.outer(np.sqrt(np.sqrt(np.reciprocal(np.array(InteractiveGraph.M_DISTS).astype(float)))), cls.ellipse_colour)
        
        @classmethod
        def remove_border(cls, ax: plt.Axes) -> None:
            ax.spines['bottom'].set_color(cls.bg_colour)
            ax.spines['top'].set_color(cls.bg_colour)
            ax.spines['right'].set_color(cls.bg_colour)
            ax.spines['left'].set_color(cls.bg_colour)


    # constant
    M_DISTS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    MAX_CUTOFF: float = None
    PRECISION: int = None
    PREPLOTS: list[np.ndarray] = None
    CALC: Calculator = None

    # variable
    curr_proj: np.ndarray = None
    curr_collection = None
    colours = None
    m_dists_using = [True, True, True, False, False, False, False, False]

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
            self.CALC = Calculator(readfile)
        else:
            self.CALC = calc
        
        self.MAX_CUTOFF = self.CALC.get_max_cutoff()*0.95
        self.PRECISION = preplots if type(preplots) is int else len(preplots)
        self.point_colours = np.vstack([self.Palette.points_colour]*len(self.CALC))
        

        self.fig = plt.figure(facecolor=self.Palette.bg_colour)
        self.ax = self.fig.add_axes([0.3, 0.1, 0.65, 0.75], facecolor = self.Palette.graph_bg)
        
        if type(preplots) is int:

            P = None
            self.PREPLOTS = []
            for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
                cutoff = i*self.MAX_CUTOFF/self.PRECISION

                u, v = self.CALC.optimise_plane(cutoff=cutoff, factor = None, from_plane=P, verbose = False)
                P = np.vstack([u, v])
                self.PREPLOTS.append(P)

        else:
            self.PREPLOTS = preplots

        self.limit = self.CALC.get_max_norm()


        self.ax.set_aspect('equal', adjustable='box')

        # Slider
        self.axslider = self.fig.add_axes([0.2, 0.1, 0.02, 0.75], facecolor=self.Palette.slider_bg)
        self.axslider.set_title("Cutoff", fontdict=self.Palette.subtitle_font)
        self.cutoff_slider = Slider(
            ax=self.axslider,
            label='',
            valmin=0,
            valmax=self.MAX_CUTOFF,
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
        self.axcheckbox = self.fig.add_axes([0.05, 0.5, 0.07, 0.35], facecolor=self.Palette.slider_bg)
        self.axcheckbox.set_title("Ellipses", fontdict=self.Palette.subtitle_font)
        self.Palette.remove_border(self.axcheckbox)

        self.m_checkbox = CheckButtons(
            ax = self.axcheckbox,
            labels = ["1", "2", "3", "4", "5", "6", "7", "8"],
            label_props={'color': self.Palette.ellipse_colours, "size":[14]*8, "family":['serif']*8},
            frame_props={'edgecolor': self.Palette.ellipse_colours, "facecolor":'white'},
            check_props={'facecolor': self.Palette.ellipse_colours},
            actives=self.m_dists_using
        )
        self.m_checkbox.on_clicked(self.show_ellipse)


        # Button
        self.axrandbutton = self.fig.add_axes([0.05, 0.4, 0.07, 0.05])
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
        self.axlassobutton = self.fig.add_axes([0.05, 0.3, 0.07, 0.05])
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

        self.update(self.PREPLOTS[0])

        self.fig.suptitle('Projected data', fontdict=self.Palette.title_font, fontsize=24)
    
    def update(self, proj: np.ndarray | None = None):
        if proj is not None:
            self.curr_proj = proj
        self.ax.cla()
        proj_points = self.curr_proj @ self.CALC.get_data()
        self.curr_collection = self.ax.scatter(proj_points[0], proj_points[1], marker = ".")
        self.curr_collection.set_facecolor(self.point_colours)
        
        #self.ax.scatter(self.curr_proj.black_points[0], self.curr_proj.black_points[1], c = [self.Palette.points_colour], marker = ".")
        ellipses = self.CALC.get_proj_ellipses(self.curr_proj, [m for m in self.M_DISTS if self.m_dists_using[m-1]])
        for i, ellipse in enumerate(ellipses):
            self.ax.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        for proj_axis in self.curr_proj.T:
            self.ax.plot([0, proj_axis[0]*self.limit*0.5], [0, proj_axis[1]*self.limit*0.5], c = self.Palette.axes_colour, linewidth = 1)
        
        self.ax.set_xbound(-self.limit, self.limit)
        self.ax.set_ybound(-self.limit, self.limit)

        self.fig.canvas.draw_idle()

    def change_cutoff(self, val = None):
        proj = self.PREPLOTS[int(self.cutoff_slider.val*self.PRECISION/self.MAX_CUTOFF)]
        ind = self.CALC.partition_data(self.cutoff_slider.val)
        self.point_colours[ind:] = self.Palette.points_colour
        self.point_colours[:ind] = self.Palette.redundant_points_colour
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
            points = self.CALC.get_data()[:, self.selector.ind]
            #print(selector.xys[selector.ind])
            self.ax.set_title("Loading projection...")
            u, v = self.CALC.optimise_plane(points=points, from_plane=self.curr_proj)
            P = np.vstack([u, v])
            self.update(P)
            
            self.selector.disconnect()
            self.fig.canvas.draw_idle()

    def show_ellipse(self, index):
        self.m_dists_using[int(index) - 1] = not self.m_dists_using[int(index) -1]
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