import pickle as pl
import numpy as np
import matplotlib.pyplot as plt
from gui import InteractiveGraph

"""
filename = input("Enter path to file: ")
preplots, calc, limits = pl.load(open(filename, 'rb'))
this_graph = InteractiveGraph(preplots, calc, limits)
plt.show()
"""
circle_res = 500

X = np.linspace(0, 6*np.pi, num=circle_res)
Y = np.linspace(0, 6*np.pi, num=circle_res)
        
circle = np.vstack([X*np.cos(X), Y*np.sin(Y), np.zeros(circle_res), np.zeros(circle_res)]).T

cov = np.identity(4)
data = None
for i, circle_point in enumerate(circle):
    group = np.random.multivariate_normal(mean=circle_point, cov=np.identity(4), size=int(200/np.sqrt(i+1)))
    if data is None:
        data = group
    else:
        data = np.vstack([data, group])

data = np.vstack([data, np.array([0, 0, 15, 0])])

"""
vec1 = np.array([1, 3, 2, -4, -2, 1])
vec2 = np.array([5, -2, 1, 0, 1, -4])
for i in range(4):
    group1 = np.random.multivariate_normal((i)*vec1, np.identity(6)/(i**3+1), 2400//(i**3+1))
    group2 = np.random.multivariate_normal((i)*vec2, np.identity(6)/(i**3+1), 2400//(i**3+1))
    if i == 0:
        data = np.vstack([group1, group2])
    else:
        data = np.vstack([data, group1, group2])
#middle = np.random.multivariate_normal(0*vec1, )
"""
np.savetxt("spiral.csv", data, delimiter=",")