print("Running...")

import matplotlib.pyplot as plt
import argparse
from interactive_graph import InteractiveGraph
from interactive_function import InteractiveFunction

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="path to independent data")
parser.add_argument("-sp", "--shared_path", type =str, default="", help = "shared path to all entries, ending with '/'")
parser.add_argument("-d", "--dependent_data", type = str, help="path to dependent data")
parser.add_argument("-ci", "--cov_independent_data", type = str, help = "path to covariance matrix for independent data")
parser.add_argument("-cd", "--cov_dependent_data", type = str, help = "path to covariance matrix for dependent data")
parser.add_argument("-mi", "--mean_independent_data", type=str, help="path to ellipse centre for independent data")
parser.add_argument("-md", "--mean_dependent_data", type=str, help="path to ellipse centre for dependent data")

args = parser.parse_args()
sp = args.shared_path

data = sp + args.data
ci = None if args.cov_independent_data is None else sp + args.cov_independent_data
mi = None if args.mean_independent_data is None else sp + args.mean_independent_data

dd = None if args.dependent_data is None else sp + args.dependent_data
cd = None if args.cov_dependent_data is None else sp + args.cov_dependent_data
md = None if args.mean_dependent_data is None else sp + args.mean_dependent_data

if args.dependent_data is None:
    this_graph = InteractiveGraph(data=data, cov_data=ci, mean_data=mi)
else:
    this_func = InteractiveFunction(data=data, dep_data=dd, cov_data=ci, mean_data=mi, cov_dep=cd, mean_dep=md)
plt.show()