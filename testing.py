import numpy as np
import matplotlib.pyplot as plt
from ellipse_projection_v1 import EllipseCalculator

EC = EllipseCalculator(6)

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


EC.set_ellipsoid(A)

EC.apply_new_rotation((1, 3), np.pi/4)
EC.apply_new_rotation((0, 5), np.pi/6)
EC.apply_new_rotation((0, 1), np.pi/3)
#print(EC)
#EC.plot_on_plane()
EC.set_axis_bias([30, 20, 10, 20], [75, 25])


