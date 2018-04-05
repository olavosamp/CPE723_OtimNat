import numpy                as np
import matplotlib.pyplot    as plt

import utils

np.set_printoptions(precision=6)#, suppress=True)

## Define clusterization problem and generate dataset
M = 10
clusterCenters = [  (0, 0),
                    (10, 10),
                    (-10,-10),
                    (10,-10),
                    (30,40)]
sigma = 1

numClusters = len(clusterCenters)
clusterShape = np.shape(clusterCenters[0])

xShape = (numClusters,) + (M,) + clusterShape
X = np.empty(xShape)
print("X shape: ", X.shape)
for index in range(numClusters):
    X[index, :, :] = sigma*np.random.standard_normal((M,) + clusterShape) + clusterCenters[index]
#
#     plt.plot(X[index, :, 1], X[index, :, 0], '.')   # Plot true centroids
#     plt.plot(clusterCenters[index][1], clusterCenters[index][0], 'x',  markersize=12)   #
#
# plt.title("Data and True Centroids")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.show()

## Optimize using Deterministic Annealing
X_all = np.reshape(X, (numClusters*M,) + clusterShape).T

# tempVec = np.arange(100, 0.001, 1)
tempVec = [100, 80, 70, 40, 10, 1]
# T = 100
epsilon = 0.001
y = np.random.random(np.shape(clusterCenters)).T    # Choose random initial centroids

for T in tempVec:
    P_yx  = utils.prob_yx(X_all, y, T)
    initD = utils.cluster_cost(P_yx, X_all, y)
    D = [0, initD]
    print("\nT = ", T)
    print("\nInitial centroids: ", y)
    print("\nInitial cost: ", D[-1])
    while(np.abs(D[-1] - D[-2]) > epsilon):
        # Conditional probabilty matrix P_y|x
        P_yx = utils.prob_yx(X_all, y, T)

        # print("P_y|x:\n", P_yx)
        print(np.sum(P_yx, 0))

        # Update centroids
        y = utils.centroid_update(P_yx, X_all, numClusters)

        # Re-calculate cost function
        D.append(utils.cluster_cost(P_yx, X_all, y))
        print("\nIteration {}. Cost: {:.5f}".format(len(D)-2, D[-1]))
        print("P_yx shape: ", P_yx.shape)

        # print("\nNew centroids: \n", y)
        # print("Cost: ", D[-1])

print("\nFinal cost: \n", D[-1])
print("\nTrue centroids\n", clusterCenters)
print("\nFinal centroids\n", y)
