import pickle as pl
import matplotlib.pyplot as plt
from gui import InteractiveGraph

filename = input("Enter path to file: ")
preplots, calc, limits = pl.load(open(filename, 'rb'))
this_graph = InteractiveGraph(preplots, calc, limits)
plt.show()