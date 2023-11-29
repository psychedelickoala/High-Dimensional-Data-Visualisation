import numpy as np
import matplotlib.pyplot as plt
from ellipse_projection_v1 import EllipseCalculator
from data_analysis_v1 import StatsCalculator

SC = StatsCalculator("sample_data.csv")
EC = EllipseCalculator(SC)

"""
test_axes = np.array(
    [
        [6, 0, 2, 1, 0, 7],
        [1, 5, 0, 3, 0, 3],
        [0, 1, 9, 4, 2, 2],
        [1, 2, 0, 6, 1, 0],
        [6, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 8, 3]
    ]
)

A = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 6]
    ]
)
"""

#EC.set_ellipsoid(SC.get_covariance())

#EC.apply_new_rotation((1, 3), np.pi/4)
#EC.apply_new_rotation((0, 5), np.pi/6)
#EC.apply_new_rotation((0, 1), np.pi)
print(EC)
#print(EC.__attribute_names)
x_bias = {"attr1": 50, "attr2": 50}
y_bias = {"attr5": 100}

EC.set_axis_bias(x_bias, y_bias)
print(EC)
EC.plot_on_plane()


