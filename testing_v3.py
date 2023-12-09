import numpy as np
from ellipse_projection_v3 import EllipseCalculator
from data_analysis_v3 import StatsCalculator
from time import time


dim = 3
num_data = 5

data = np.random.rand(dim, num_data) - 0.5
avging_vec = np.ones((num_data, 1))/num_data

SC = StatsCalculator(data)


