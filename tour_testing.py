import numpy as np
from calculator import Calculator

from langevitour import Langevitour

# Generate a sample dataset
calc = Calculator("sample_data.csv")
X = calc.get_data().T

n = 20000
group = [0]*n

# Extra axes (specified as columns of a matrix)
#extra_axes = [[1], [2], [0], [0], [0]]
#extra_axes_names = ["V1+2*V2"]

tour = Langevitour(
    X,
    #group=group,
    #extra_axes=extra_axes,
    #extra_axes_names=extra_axes_names,
    point_size=1,
)
tour.write_html("langevitour_plot.html")