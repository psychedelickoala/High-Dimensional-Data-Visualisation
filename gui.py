import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from calculator import Calculator, ProjGraph
from matplotlib.widgets import Button, Slider, CheckButtons
import pickle as pl

# Keeping track of interactive elements
class InteractiveGraph:

    # Colours
    class Palette:
        light_green = [0.898, 0.988, 0.761]
        green = [0.616, 0.878, 0.678]
        blue = [0.271, 0.678, 0.659]
        dark_blue = [0.329, 0.475, 0.502]
        grey = [0.349, 0.31, 0.31]
        off_white = [0.969, 0.945, 0.929]
        black = [0, 0, 0]

        ellipse_colour = blue
        ellipse_colours = None
        bg_colour = green
        slider_colour = dark_blue
        slider_bg = off_white + [0.5]
        graph_bg = off_white
        points_colour = grey
        redundant_points_colour = [rgb * 0.9 for rgb in graph_bg]
        axes_colour = black

        title_font = {"color": off_white, "family": "serif"}
        subtitle_font = {"color": off_white, "family": "serif", "size": 14}

        @classmethod
        def __init__(cls) -> None:
            cls.ellipse_colours = [[rgb * np.sqrt(np.sqrt(1/m_dist)) for rgb in cls.ellipse_colour] for m_dist in InteractiveGraph.M_DISTS]
        
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
    PREPLOTS: list[ProjGraph] = None
    CALC: Calculator = None

    # variable
    curr_proj: ProjGraph = None
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

    def __init__(self, preplots: list[ProjGraph] | int = 200, calc: Calculator = None, limit: float = None) -> None:
        self.Palette()
        if calc is None:
            readfile = input("Enter path to data csv file: ")
            self.CALC = Calculator(readfile)
        else:
            self.CALC = calc
        self.MAX_CUTOFF = self.CALC.get_max_cutoff()*0.95
        self.PRECISION = preplots if type(preplots) is int else len(preplots)
        

        self.fig = plt.figure(facecolor=self.Palette.bg_colour)
        self.ax = self.fig.add_axes([0.3, 0.1, 0.65, 0.75], facecolor = self.Palette.graph_bg)
        
        if type(preplots) is int:

            P = None
            self.PREPLOTS = []
            for i in tqdm(range(self.PRECISION), desc = "Finding projections..."):
                cutoff = i*self.MAX_CUTOFF/self.PRECISION

                u, v = self.CALC.optimise_plane(cutoff=cutoff, factor = None, from_plane=P, verbose = False)
                P = np.vstack([u, v])
                self.PREPLOTS.append(ProjGraph(P, self.CALC, cutoff, self.M_DISTS))

        else:
            self.PREPLOTS = preplots
            ProjGraph.lim = limit

        self.limit = ProjGraph.lim

        self.update(self.PREPLOTS[0])
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

        self.fig.suptitle('Projected data', fontdict=self.Palette.title_font, fontsize=24)
    
    def update(self, proj: ProjGraph | None = None):
        if proj is not None:
            self.curr_proj = proj
        self.ax.cla()
        self.ax.scatter(self.curr_proj.grey_points[0], self.curr_proj.grey_points[1], c = [self.Palette.redundant_points_colour], marker = ".")
        self.ax.scatter(self.curr_proj.black_points[0], self.curr_proj.black_points[1], c = [self.Palette.points_colour], marker = ".")

        for i, ellipse in enumerate(self.curr_proj.ellipses):
            if self.m_dists_using[i]:
                self.ax.plot(ellipse[0], ellipse[1], c=self.Palette.ellipse_colours[i], linewidth = 2)

        for proj_axis in self.curr_proj.P.T:
            self.ax.plot([0, proj_axis[0]*self.limit*0.4], [0, proj_axis[1]*self.limit*0.4], c = self.Palette.axes_colour, linewidth = 1)
        
        self.ax.set_xbound(-self.limit*1.1, self.limit*1.1)
        self.ax.set_ybound(-self.limit*1.1, self.limit*1.1)

        self.fig.canvas.draw_idle()

    def change_cutoff(self, val):
        proj = self.PREPLOTS[int(val*self.PRECISION/self.MAX_CUTOFF)]
        self.update(proj)

    def show_random(self, event):
        P = self.CALC.get_random_plane()
        rand_proj = ProjGraph(P, self.CALC, self.cutoff_slider.val, self.M_DISTS, False)
        self.update(rand_proj)

    def show_ellipse(self, index):
        self.m_dists_using[int(index) - 1] = not self.m_dists_using[int(index) -1]
        self.update()

# Save to binary file
if __name__=='__main__':
    this_graph = InteractiveGraph(200)
    #writefile = input("Enter path to save widget (.pickle extension): ")
    #pl.dump((this_graph.PREPLOTS, this_graph.CALC, this_graph.limits), open(writefile,'wb'))
    plt.show()