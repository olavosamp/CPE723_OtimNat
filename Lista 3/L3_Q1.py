# Lista 2
# Questao 2
#
# CPE723
# Olavo Sampaio

import numpy                as np
import matplotlib.pyplot    as plt
# import random

import utils

np.set_printoptions(precision=3, suppress=True)

## Letra A

# X = np.ndarray([0, 4, 6, 9], dtype=int)
X = [0, 4, 6, 9]
t = np.arange(-1.0, 10.0, 0.1)
D = np.empty(len(t))
i = 0
for partition in t:
    X1 = [elem for elem in X if elem <= partition]
    X2 = [elem for elem in X if elem > partition]

    y1 = utils.center_mass(X1)
    y2 = utils.center_mass(X2)

    dist1 = np.sum(utils.euclidean_dist(X1, y1))
    dist2 = np.sum(utils.euclidean_dist(X2, y2))

    D[i] = (dist1+dist2)/(len(X1)+len(X2))

    # print("\ni: {}, t : {:.1f}\nD: {:.3f}".format(i, partition, D[i]))
    i += 1

plt.plot(t, D)
# plt.set_title("Gráfico de D(t)")
plt.ylabel("D(t)")
plt.xlabel("t")
plt.show()

## Letra B
