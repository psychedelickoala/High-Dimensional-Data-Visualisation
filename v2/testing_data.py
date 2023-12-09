import numpy as np
from ellipse_projection_v2 import EllipseCalculator
from data_analysis_v2 import StatsCalculator

dim = 3
num_data = 5

data = np.random.rand(dim, num_data) - 0.5
avging_vec = np.ones((num_data, 1))/num_data

SC = StatsCalculator(data)
'''
mean = data @ avging_vec
print(mean)

A = np.cov(data)
AI = np.linalg.inv(A)

u = np.random.rand(dim) - 0.5
v = np.random.rand(dim) - 0.5

EC = EllipseCalculator(A)
ECI = EllipseCalculator(AI)
P = EC.get_projection_matrix(u, v)

ellipses = EC.ellipsoid_onto_plane(P, mean, [1, 2, 3])
ellipsesI = ECI.ellipsoid_onto_plane(P, mean, [1, 2, 3])
projected_data = EC.points_onto_plane(P, data)
#axes = EC.axes_onto_plane(P)

EC.plot_on_plane(ellipses, projected_data)
ECI.plot_on_plane(ellipsesI, projected_data)'''