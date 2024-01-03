import numpy as np
from ellipse_projection_v2 import EllipseCalculator
from time import time

# In this testing program we are measuring efficiency.

# n is the number of dimensions of our hyperellipsoid
n: int = 6

# trials100 is the number of lots of 100 trials we are runningq
trials100: int = 10

# num_points is the number of points we want each time we make an ellipse
num_points: int = 50

num_data_points = 50

start = time()

print(f"Projecting a {n}x{n} hyperellipsoid, {trials100 * 100} times")

B = np.random.rand(n, n)
A = B @ B.T
EC = EllipseCalculator(A, num_points)

# Running trials
for i in range(trials100 * 100):
    if i%(10*trials100) == 0:
        print(f"Generating points, {i//(trials100)}% complete")
    # u and v are vectors spanning the plane we want to project onto. They do not have to be orthonormal
    u = np.random.rand(n)
    v = np.random.rand(n)
    P = EC.get_projection_matrix(u, v)
    # gets 2 x n array of points on the ellipse
    mean = 10*np.random.rand(n)
    ellipses = EC.ellipsoid_onto_plane(P, mean, [1, 2, 3, 4])

end = time()

print(f"Execution complete. Program ran in {end - start} seconds.")

# include data points
data = (5*np.random.rand(n, num_data_points) - 2.5) + np.tile(mean, (num_data_points, 1)).T
projected_data = EC.points_onto_plane(P, data)
print(f"Projected data: \n {projected_data}")

# displaying the last ellipse generated.
print(f"Last of {trials100*100} trials:")
print(f"Hyperellipsoid: \n {np.round(A, 2)}")
print(f"Plane spanned by \n u = {np.round(u, 2)},\n v = {np.round(v, 2)}")
print(f"Points: \n {np.round(ellipses, 1)}")
EC.plot_on_plane(ellipses, projected_data)


