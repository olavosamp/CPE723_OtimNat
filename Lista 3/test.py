import numpy                as np
import matplotlib.pyplot    as plt

import utils

x = np.array([[0, 4, 6, 9], [0, 4, 6, 9]])
y = np.array([[3.0, 3.4], [3.0, 3.4]])

print(x)
print(y)

dist1 = np.sum((x[:, 0] - y[:, 0])**2)
distFunc = utils.squared_dist(x[:, 0], y[:, 0])
print(dist1)
print(distFunc)

pxy = utils.prob_yx(x, y, 1)
print(pxy)
print(pxy.sum(axis=1))
