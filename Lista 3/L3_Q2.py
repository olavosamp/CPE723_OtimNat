import numpy                as np
import matplotlib.pyplot    as plt

import utils

M = 1000
clusterCenters = [  (0, 0),
                    (1, 1),
                    (-1,-1),
                    (1,-1),
                    (-1,1)]
sigma = 0.3

numClusters = len(clusterCenters)
clusterShape = np.shape(clusterCenters[0])

xShape = (numClusters,) + (M,) + clusterShape
X = np.empty(xShape)
print("X shape: ", X.shape)
for index in range(numClusters):
    X[index, :, :] = sigma*np.random.standard_normal((M,) + clusterShape) + clusterCenters[index]

    plt.plot(X[index, :, 1], X[index, :, 0], '.')   # Plot true centroids
    plt.plot(clusterCenters[index][1], clusterCenters[index][0], 'x',  markersize=15)   #

    # print("X{} std: {}".format(index, np.std(X[index], axis=0)))

# print("\nX1 mean: {}\nX1 std: {}".format(np.mean(X1), np.std(X1)))
# print("\nX2 mean: {}\nX2 std: {}".format(np.mean(X2), np.std(X2)))

plt.show()
