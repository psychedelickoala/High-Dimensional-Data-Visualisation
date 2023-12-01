import numpy as np
import matplotlib.pyplot as plt
from ellipse_projection_v2 import EllipseCalculator
from data_analysis_v2 import StatsCalculator
from time import time

# In this testing program we are measuring efficiency.

# n is the number of dimensions of our hyperellipsoid
n = 12

# trials100 is the number of lots of 100 trials we are running
trials100 = 10

# num_points is the number of points we want each time we make an ellipse
num_points = 50

start = time()

print(f"Projecting a {n}x{n} hyperellipsoid, {trials100 * 100} times")

B = np.random.rand(n, n)
A = B @ B.T
EC = EllipseCalculator(A)

# Running trials
for i in range(trials100 * 100):
    if i%(10*trials100) == 0:
        print(f"Generating points, {i//(trials100)}% complete")
    # u and v are vectors spanning the plane we want to project onto. They do not have to be orthonormal
    u = np.random.rand(n)
    v = np.random.rand(n)
    # T is a 2x2 matrix describing the ellipse as a transformation of a circle
    T = EC.project_onto_plane(u, v)
    # points is a 2 x num_points array of all the points on the projected ellipse
    points = EC.transformation_to_points(T, num_points=num_points)

end = time()

print(f"Execution complete. Program ran in {end - start} seconds.")

# displaying the last ellipse generated.
print(f"Last of {trials100*100} trials:")
print(f"Hyperellipsoid: \n {np.round(A, 2)}")
print(f"Plane spanned by \n u = {np.round(u, 2)},\n v = {np.round(v, 2)}")
print(f"Transform each point on the circle by \n {np.round(T, 2)}")
EC.plot_on_plane(points)


