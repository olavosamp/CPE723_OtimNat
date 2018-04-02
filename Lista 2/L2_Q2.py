# Lista 2
# Questao 2
#
# CPE723
# Olavo Sampaio

import numpy                as np
import matplotlib.pyplot    as plt
import random

from utils import get_random_event, perturb

# Pretty printing options
np.set_printoptions(precision=3, suppress=True)

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

# # Letra E
# # Executar SA usando a matriz de transição
#
# # Vetor de temperaturas
# # T = [0.1000, 0.0631, 0.0500, 0.0431, 0.0387, 0.0356, 0.0333, 0.0315, 0.0301, 0.0289]
# iters = 1000    # Iteracoes
# N = 1000    # Tamanho do conj de dados
# t = 5       # Num de Estados
# T = 0.1
#
# X = np.empty((N, iters))
# X[:, 0] = np.random.randint(0, 5, size=N)  # Inicializar X[0]
#
# for i in range(0, iters+1):
#     # Perturbation of X[i]
#     x_new = perturb(X[:,i], t)
#     X[:, i+1] = np.random.choice(t, size=N, p=)
#     X[:,i+1] = get_random_event(M[x_new,:], x_new)
#
#     # if np.mod(i, M) == 0:
#     #     T = T_0/(np.log2(1 + k))
#     #     k = k+1
#
# bins = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2]
# plt.hist(X[0, :], bins=bins)
# plt.set_title("Distribuição final para T = ", T)
# plt.set_ylabel("Contagem")
# plt.hist(X[-1, :], bins=bins)
