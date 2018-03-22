# Lista 2
# Questao 2
#
# CPE723
# Olavo Sampaio

import numpy                as np
import matplotlib.pyplot    as plt
import random

from utils import get_random_event  # Same function from Q1

# np.set_printoptions(precision=3)

# Letra A
M =[[0,     np.exp(-3),     np.exp(-2),     np.exp(-4),              np.exp(-1)],
    [1,     0,              1,              np.exp(-1),     1],
    [1,     np.exp(-1),     0,              np.exp(-2),     1],
    [1,     1,              1,              0,              1],
    [1,     np.exp(-2),     np.exp(-1),     np.exp(-3),     0]
]

M = np.divide(M,5)
for i in range(len(M)):
    M[i,i] = 1 - np.sum(M[:,i])

for i in range(len(M)):
    M[:, i] = np.divide(M[:,i],np.sum(M[:,i]))



print("\nLetra A\nMatriz M\n")
print(M)

# Letra B

# Print probabilty decision boundaries for each state
print("\nLetraB\nLimiares de decisao\n")
edge = np.zeros(6)
for i in range(len(M)):
    edge[0] = 0
    for j in range(0, len(M)):
        # print("M[{},{}]: {}".format(i, j, np.around(M[j, i], 3)))
        edge[j+1] = edge[j] + M[j, i]
    print("\nP_{}: 0---{:.5f}---{:.3f}---{:.3f}---{:.3f}---{:.3f}".format(i, edge[1], edge[2], edge[3], edge[4], edge[5]))

# Letra C
w, v = np.linalg.eig(M)

piVec = np.divide(v[:, 0], np.sum(v[:,0]))

print("\nLetra C\n")
print("\nAutovalor 0: ", w[0])
print("\nAutovetor 0 normalizado (é o vetor invariante, Pi): ")
print(piVec)

print("\nVetor Pi deve somar 1: ", np.sum(piVec))

# Letra D
J = [0.5, 0.2, 0.3, 0.1, 0.4]
T = 0.1

boltz = lambda x: np.exp(-x/T)

fatoresBoltz = list(map(boltz, J))
fatoresBoltz = np.divide(fatoresBoltz, np.sum(fatoresBoltz))

print("\nLetra D\nComparação com Fatores de Boltzmann\n")
print("Fatores de Boltz normalizados\n", fatoresBoltz)
print("\nVetor invariante\n", piVec)
print("\nDiferença\n", np.abs(fatoresBoltz-piVec))
