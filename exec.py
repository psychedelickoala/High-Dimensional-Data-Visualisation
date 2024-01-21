import matplotlib.pyplot as plt
import argparse
import pickle as pl
from gui import InteractiveGraph

 # parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="relative path to independent data")
parser.add_argument("-sp", "--shared_path", type =str, default="", help = "shared path to all entries, ending with '/'")
parser.add_argument("-d", "--dependent_data", type = str, help="relative path to dependent data")
parser.add_argument("-ci", "--cov_independent_data", type = str, help = "relative path to covariance matrix for independent data")
parser.add_argument("-cd", "--cov_dependent_data", type = str, help = "relative path to covariance matrix for dependent data")
parser.add_argument("-mi", "--mean_independent_data", type=str, help="relative path to ellipse centre for independent data")
parser.add_argument("-md", "--mean_dependent_data", type=str, help="relative path to ellipse centre for dependent data")

args = parser.parse_args()
print(args)
#this_graph = InteractiveGraph()
#writefile = input("Enter path to save widget (.pickle extension): ")
#pl.dump((this_graph.PREPLOTS, this_graph.CALC, this_graph.limits), open(writefile,'wb'))
#plt.show()