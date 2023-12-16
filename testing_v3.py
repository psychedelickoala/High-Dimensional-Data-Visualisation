import numpy as np
from ellipse_projection_v3 import EllipseCalculator
from data_analysis_v3 import StatsCalculator
import matplotlib.pyplot as plt


dim = 5
num_data = 10000
conf = 3
goof_dist = 20

#data = np.random.chisquare(3, size = (dim, num_data))
#data = np.random.rand(dim, num_data)

cov_base = np.random.rand(dim, dim)
cov = cov_base @ cov_base.T
centre = np.random.rand(dim)

data = np.random.multivariate_normal(centre, cov, size=num_data)

goofy_data = np.random.multivariate_normal(goof_dist*np.random.rand(dim), 0.1*cov, size=num_data)
#goofy_data = np.random.chisquare(3, size = (dim, num_data)).T
data = np.vstack([data, goofy_data]).T

SC = StatsCalculator(data)
K = SC.get_outlier_plane(conf)

M = K.T @ K

C = SC.get_covariance()

EC = EllipseCalculator(np.linalg.inv(C))

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
    #return (u @ C @ u.T - u @ M @ u.T)**2 + (v @ C @ v.T - v @ M @ v.T)**2  + (v @ C @ u.T)**2
    return (u @ C @ u.T * u @ M @ u.T - 1)**2 + (v @ C @ v.T * v @ M @ v.T - 1)**2  + (v @ C @ u.T)**2   # + (v @ u.T)**2 + (v @ v.T - 1)**2 + (u @ u.T - 1)**2

def grad(u, v):
    global C, M
    
    du = 4 * (u @ C @ u.T * u @ M @ u.T - 1) * ((u @ C @ u.T) * M @ u.T + (u @ M @ u.T)* C @ u.T) + 2 * (u @ C @ v.T) * C @ v.T  #  +  2 * (u @ v.T) * v.T + 4 * (u @ u.T - 1) * u.T
    #du = 4*(u @ C @ u.T - u @ M @ u.T)*(C @ u.T - M @ u.T) + 2 * (u @ C @ v.T) * C @ v.T
    dv = 4 * (v @ C @ v.T * v @ M @ v.T - 1) * ((v @ C @ v.T) * M @ v.T + (v @ M @ v.T)* C @ v.T) + 2 * (v @ C @ u.T) * C @ u.T  #  + 2 * (v @ u.T) * u.T  + 4 * (v @ v.T - 1) * v.T
    #dv = 4*(v @ C @ v.T - v @ M @ v.T)*(C @ v.T - M @ v.T) + 2 * (v @ C @ u.T) * C @ v.T
    return du / np.linalg.norm(du)**2, dv / np.linalg.norm(dv)**2

def grad_descent(u, v, tol = 1e-8, verbose = False):
    i = 1
    u, v = orthonormalise(u, v)
    c = f(u, v)
    while c > tol:
        g = grad(u, v)
        prev_u, prev_v = u.copy(), v.copy()
        u, v = orthonormalise(prev_u - c*g[0], prev_v - c*g[1])
        if (np.allclose(u, prev_u) and np.allclose(v, prev_v)) or i > 100000:
             raise TimeoutError(f"Error converged to {c}")
        c = f(u, v)
        if verbose:
            print(f"\n\ntrial {i}:\n grad = \n{g}\nu = \n{u}\nv = \n{v}\nf(u) = \n{c}")
        i += 1
        
    if verbose:
        print(f"Process completed in {i} steps")
    return u, v

u = np.random.rand(dim) - 0.5
v = np.random.rand(dim) -0.5
u, v = grad_descent(u, v, tol = 0.2)
P = np.vstack([u, v])




#print(best_P)
#print(np.abs(P.T @ np.linalg.inv(P @ C @ P.T) @ P - K.T @ K))



#EC = EllipseCalculator(A, num_points=30)

#P = EC.get_projection_matrix(u, v)
proj_data_opt = EC.points_onto_plane(P, SC.get_data())
axes_opt = EC.axes_onto_plane(P)
ellipses_opt = EC.ellipsoid_onto_plane(P, SC.get_mean(), m_dists = [conf, 1, 2, 3])



plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#fecb3e", "#fc8370", "#c2549d", "#7e549e"])

fig, axs = plt.subplots(2, 2)
#axs[0, 0].plot(x, y)
opt = axs[0, 0]
opt.set_title('Optimised')
#axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Random 1')
#axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Random 2')
#axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Random 3')

rand_plots = [axs[0, 1], axs[1, 0], axs[1, 1]]

for i in range(dim):
    opt.plot([0, axes_opt[i][0]], [0, axes_opt[i][1]], c = "grey", linewidth = 1)

for i in range(len(ellipses_opt)):
    opt.plot(ellipses_opt[i][0], ellipses_opt[i][1])

opt.scatter(proj_data_opt[0], proj_data_opt[1], c = "#000000", marker = ".")


for rplot in rand_plots:
    P = EC.get_projection_matrix(np.random.rand(dim) - 0.5, np.random.rand(dim) - 0.5)
    proj_data = EC.points_onto_plane(P, SC.get_data())
    axes = EC.axes_onto_plane(P)
    ellipses = EC.ellipsoid_onto_plane(P, SC.get_mean(), m_dists = [conf, 1, 2, 3])

    for i in range(dim):
        rplot.plot([0, axes[i][0]], [0, axes[i][1]], c = "grey", linewidth = 1)

    for i in range(len(ellipses)):
        rplot.plot(ellipses[i][0], ellipses[i][1])

    rplot.scatter(proj_data[0], proj_data[1], c = "#000000", marker = ".")

plt.show()

