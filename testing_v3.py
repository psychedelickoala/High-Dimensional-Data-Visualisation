import numpy as np
from ellipse_projection_v3 import EllipseCalculator
from data_analysis_v3 import StatsCalculator
from time import time


dim = 3
num_data = 1000

data = np.random.chisquare(4, size = (dim, num_data))

#cov_base = np.random.rand(dim, dim)
#cov = cov_base @ cov_base.T
#centre = np.random.rand(dim)

#data = np.random.multivariate_normal(centre, cov, size=num_data)

SC = StatsCalculator(data)
u, v = SC.get_outlier_plane(3)

A = np.linalg.inv(SC.get_covariance())
EC = EllipseCalculator(A, num_points=30)

P = EC.get_projection_matrix(u, v)
proj_data = EC.points_onto_plane(P, SC.get_data())
axes = EC.axes_onto_plane(P)
ellipses = EC.ellipsoid_onto_plane(P, SC.get_mean(), m_dists = [1, 2, 3])

EC.plot_on_plane(ellipses, proj_data, axes)

