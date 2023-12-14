import numpy as np
from ellipse_projection_v3 import EllipseCalculator
from data_analysis_v3 import StatsCalculator
from time import time


dim = 6
num_data = 100000

data = np.random.chisquare(3, size = (dim, num_data))
#data = np.random.rand(dim, num_data)

cov_base = np.random.rand(dim, dim)
cov = cov_base @ cov_base.T
centre = np.random.rand(dim)

#data = np.random.multivariate_normal(centre, cov, size=num_data).T

SC = StatsCalculator(data)
K = SC.get_outlier_plane(3.5)

M = K.T @ K

C = SC.get_covariance()

EC = EllipseCalculator(C)

B = np.linalg.cholesky(C).T

def orthonormalise(vec1, vec2):
        # orthonormalise u, v
        u = vec1 / np.linalg.norm(vec1)
        v = vec2 - np.dot(vec2, u)*u
        
        # checking linear independence
        if np.linalg.norm(v) == 0:
            raise ValueError("Matrix is low rank")
        
        v /= np.linalg.norm(v)

        # return 2 x n array
        return u, v

def f(u, v):
    global C, M
    return (u @ C @ u.T * u @ M @ u.T - 1)**2 + (v @ C @ v.T * v @ M @ v.T - 1)**2  + (v @ C @ u.T)**2   # + (v @ u.T)**2 + (v @ v.T - 1)**2 + (u @ u.T - 1)**2

def grad(u, v):
    global C, KM
    
    du = 4 * (u @ C @ u.T * u @ M @ u.T - 1) * ((u @ C @ u.T) * M @ u.T + (u @ M @ u.T)* C @ u.T) + 2 * (u @ C @ v.T) * C @ v.T  #  +  2 * (u @ v.T) * v.T + 4 * (u @ u.T - 1) * u.T
    
    dv = 4 * (v @ C @ v.T * v @ M @ v.T - 1) * ((v @ C @ v.T) * M @ v.T + (v @ M @ v.T)* C @ v.T) + 2 * (v @ C @ u.T) * C @ u.T  #  + 2 * (v @ u.T) * u.T  + 4 * (v @ v.T - 1) * v.T

    return du / np.linalg.norm(du)**2, dv / np.linalg.norm(dv)**2

def grad_descent(u, v, tol = 1e-8, verbose = False):
    i = 1
    u, v = orthonormalise(u, v)
    c = f(u, v)
    while c > tol:
        g = grad(u, v)
        prev_u, prev_v = u.copy(), v.copy()
        u, v = orthonormalise(u - c*g[0], v - c*g[1])
        if np.allclose(u, prev_u) and np.allclose(v, prev_v):
             raise TimeoutError(f"Error converged to {c}")
        c = f(u, v)
        if verbose:
            print(f"\n\ntrial {i}:\n grad = \n{g}\nu = \n{u}\nv = \n{v}\nf(u) = \n{c}")
        i += 1
        
    if verbose:
        print(f"Process completed in {i} steps")
    return u, v

u = np.random.rand(dim)
v = np.random.rand(dim)
u, v = grad_descent(u, v)
print(u, v)


#print(best_P)
#print(np.abs(P.T @ np.linalg.inv(P @ C @ P.T) @ P - K.T @ K))







#EC = EllipseCalculator(A, num_points=30)

#P = EC.get_projection_matrix(u, v)
#proj_data = EC.points_onto_plane(P, SC.get_data())
#axes = EC.axes_onto_plane(P)
#ellipses = EC.ellipsoid_onto_plane(P, SC.get_mean(), m_dists = [1, 2, 3])

#EC.plot_on_plane(ellipses, proj_data, axes)

