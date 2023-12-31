import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from calculator import Calculator
from matplotlib.widgets import Button, Slider, CheckButtons

class ProjGraph:
    max_x = -np.inf
    min_x = np.inf

    max_y = -np.inf
    min_y = np.inf

    def __init__(self, P: np.ndarray, calc: Calculator, cutoff: float, m_dists: list[float] = [1, 2, 3], remember = True) -> None:
        self.P = P
        #self.calc = calc
        self.cutoff = cutoff

        self.ellipses = calc.get_proj_ellipses(P, m_dists)
        grey_data, black_data = calc.partition_data(cutoff)
        self.grey_points, self.black_points = P @ grey_data, P @ black_data

        # get min, max
        if remember:
            min_x = np.min(self.black_points[0])
            max_x = np.max(self.black_points[0])
            min_y = np.min(self.black_points[1])
            max_y = np.max(self.black_points[1])
            ProjGraph.min_x = min_x if min_x < ProjGraph.min_x else ProjGraph.min_x
            ProjGraph.max_x = max_x if max_x > ProjGraph.max_x else ProjGraph.max_x
            ProjGraph.min_y = min_y if min_y < ProjGraph.min_y else ProjGraph.min_y
            ProjGraph.max_y = max_y if max_y > ProjGraph.max_y else ProjGraph.max_y

class Palette:
    light_green = [0.898, 0.988, 0.761]
    green = [0.616, 0.878, 0.678]
    blue = [0.271, 0.678, 0.659]
    dark_blue = [0.329, 0.475, 0.502]
    grey = [0.349, 0.31, 0.31]

    ellipse_colour = blue
    bg_colour = green
    slider_colour = dark_blue
    slider_bg = [1, 1, 1, 0.5]
    graph_bg = [0.969, 0.945, 0.929]
    points_colour = grey
    redundant_points_colour = [rgb * 0.9 for rgb in graph_bg]
    axes_colour = dark_blue

    @classmethod
    def darken(cls, colour, factor):
        return [rgb * factor for rgb in colour]

    @classmethod
    def get_ellipse_colours(self, m_dists = [1, 2, 3]) -> list[list]:
        return [Palette.darken(Palette.ellipse_colour, np.sqrt(np.sqrt(1/m_dist))) for m_dist in m_dists]




calc = Calculator("sample_data.csv")

# Define initial parameters
max_cutoff = calc.get_max_cutoff()*0.95
precision = 200
m_dists = [1, 2, 3, 4, 5, 6, 7, 8]
m_dists_using = [True, True, True, False, False, False, False, False]
preplots = []
ellipse_colours = Palette.get_ellipse_colours(m_dists)
curr_proj = None

from_plane = None
for i in tqdm(range(precision), desc = "Finding projections..."):
    print()
    cutoff = i*max_cutoff/precision

    u, v = calc.optimise_plane(cutoff=cutoff, factor = None, from_plane=from_plane, verbose = False)
    P = np.vstack([u, v])
    preplots.append(ProjGraph(P, calc, cutoff, m_dists))

    from_plane = (u, v)


def change_cutoff(val):
    proj = preplots[int(val*precision/max_cutoff)]
    update(proj)

def update(proj: ProjGraph):
    global ax, curr_proj
    curr_proj = proj
    ax.cla()
    #proj = preplots[int(val*precision/max_cutoff)]
    ax.scatter(proj.grey_points[0], proj.grey_points[1], c = [Palette.redundant_points_colour], marker = ".")
    ax.scatter(proj.black_points[0], proj.black_points[1], c = [Palette.points_colour], marker = ".")

    for i, ellipse in enumerate(proj.ellipses):
        if m_dists_using[i]:
            ax.plot(ellipse[0], ellipse[1], c=ellipse_colours[i], linewidth = 2)

    for proj_axis in proj.P.T:
        ax.plot([0, proj_axis[0]], [0, proj_axis[1]], c = Palette.axes_colour, linewidth = 1)
    
    ax.set_xbound(ProjGraph.min_x-1, ProjGraph.max_x+1)
    ax.set_ybound(ProjGraph.min_y-1, ProjGraph.max_y+1)
    fig.canvas.draw_idle()

def show_random(event):
    global ax
    #ax.cla()
    P = calc.get_random_plane()
    rand_proj = ProjGraph(P, calc, cutoff_slider.val, m_dists, False)
    update(rand_proj)


def show_ellipse(index):
    global curr_proj
    m_dists_using[int(index) - 1] = not m_dists_using[int(index) -1]
    update(curr_proj)


title_font = {"color": [1, 1, 1], "family": "serif"}
subtitle_font = {"color": [1, 1, 1], "family": "serif", "size": 14}

# Create the figure and the line that we will manipulate
fig = plt.figure(facecolor=Palette.bg_colour)
ax = fig.add_axes([0.3, 0.1, 0.65, 0.75], facecolor = Palette.graph_bg)

# adjust the main plot to make room for the sliders
#fig.subplots_adjust(left=0.3)

# Make a vertical slider to control the frequency.
axslider = fig.add_axes([0.2, 0.1, 0.02, 0.75], facecolor=Palette.slider_bg)
cutoff_slider = Slider(
    ax=axslider,
    label='',
    valmin=0,
    valmax=max_cutoff,
    valinit=0,
    handle_style={"edgecolor":Palette.slider_colour, "facecolor": Palette.graph_bg},
    orientation="vertical",
    color=Palette.slider_colour,
    track_color=Palette.slider_bg,
    closedmax=False,
    initcolor=None
)

axcheckbox = fig.add_axes([0.05, 0.5, 0.07, 0.35], facecolor=Palette.slider_bg)
m_checkbox = CheckButtons(
    ax = axcheckbox,
    labels = ["1", "2", "3", "4", "5", "6", "7", "8"],
    label_props={'color': ellipse_colours, "size":[14]*8, "family":['serif']*8},
    frame_props={'edgecolor': ellipse_colours, "facecolor":'white'},
    check_props={'facecolor': ellipse_colours},
    actives=m_dists_using
)

axcheckbox.spines['bottom'].set_color(Palette.bg_colour)
axcheckbox.spines['top'].set_color(Palette.bg_colour)
axcheckbox.spines['right'].set_color(Palette.bg_colour)
axcheckbox.spines['left'].set_color(Palette.bg_colour)

m_checkbox.on_clicked(show_ellipse)
axcheckbox.set_title("Ellipses", fontdict=subtitle_font)
axslider.set_title("Cutoff", fontdict=subtitle_font)

axrandbutton = fig.add_axes([0.05, 0.1, 0.07, 0.05])
rand_button = Button(
    ax=axrandbutton,
    label = "Random",
    color = Palette.slider_colour,
    hovercolor = Palette.blue
)
cutoff_slider.label.set_color('white')
rand_button.label.set_color('white')
rand_button.label.set_font('serif')
rand_button.label.set_fontsize(14)

axrandbutton.spines['bottom'].set_color(Palette.bg_colour)
axrandbutton.spines['top'].set_color(Palette.bg_colour)
axrandbutton.spines['right'].set_color(Palette.bg_colour)
axrandbutton.spines['left'].set_color(Palette.bg_colour)

rand_button.on_clicked(show_random)
# register the update function with each slider
cutoff_slider.on_changed(change_cutoff)

update(preplots[0])


#fig.subplots_adjust(top = 0.1)
#axtitle = fig.add_axes([0.1, 0.1, 0.9, 0.1])
fig.suptitle('Projected data', fontdict=title_font, fontsize=24)
ax.set_aspect('equal', adjustable='box')

plt.show()