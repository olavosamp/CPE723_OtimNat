# Lista 2
# Questao 2
#
# CPE723
# Olavo Sampaio

import numpy                as np
import matplotlib.pyplot    as plt
import random

from utils import get_random_event, perturb, metropolis_prob

# Pretty printing options
np.set_printoptions(precision=3, suppress=True)

# Letra A
M =[[0,     np.exp(-3),     np.exp(-2),     np.exp(-4),     np.exp(-1)],
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

# Letra E
# Executar SA usando a matriz de transição

# Funcao custo, Dom =  {0, 1, 2, 3, 4}
J_func = lambda x: J[x]

# Vetor de temperaturas
tempVec = [0.1000, 0.0631, 0.0500, 0.0431, 0.0387, 0.0356, 0.0333, 0.0315, 0.0301, 0.0289]
# tempVec = [0.1]
iters = 1000        # Iteracoes
N = 1000            # Tamanho do conj de dados
numStates = 5       # Num de Estados
bins = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2]

# T = 0.1

X = np.empty((N, iters, len(tempVec)), dtype=int)
X[:, 0, 0] = np.random.randint(0, numStates, size=N)  # Inicializar X[0]

for temp in range(len(tempVec)):
    T = tempVec[temp]
    print("Temp: ", T)
    for i in range(0, iters-1):
        # Perturbation of X[i]
        # randPos = np.random.randint(0, N)
        # newVal = np.random.randint(0, numStates)
        #
        # oldVal = X[randPos, i, temp]
        #

        xNew = np.random.randint(0, numStates, size=N)
        J_New = list(map(J_func, xNew))
        J_Old = list(map(J_func, X[:, i, temp]))
        diff = np.subtract(J_New, J_Old) # J_new - J_old elementwise

        for elem in range(N):
            randomNum = random.random()
            acceptProb = metropolis_prob(J_Old[elem], J_New[elem], T)

            if randomNum < acceptProb:
                X[elem, i+1, temp] = xNew[elem]
                # print("Estado aceito")
            else:
                X[elem, i+1, temp] = X[elem, i, temp]
                # print("Estado rejeitado")

        # print("\nJ_Old: {}\nJ_New: {}".format(J[oldVal], J[newVal]))
        # print("Accept prob: {}".format(acceptProb))
        # print("Random number: {:.3f}".format(randomNum))
        # randomNum = random.random()
        # acceptProb = metropolis_prob(J[oldVal], J[newVal], T)
        #
        #
        # if randomNum < acceptProb:
        #     X[:, i+1, temp] = xNew
        #     # print("Estado aceito")
        # else:
        #     X[:, i+1, temp] = X[:, i, temp]
        #     # print("Estado rejeitado")

    # if (temp+1) < len(tempVec):
    #     X[:, 0, temp+1] = X[:, i+1, temp]

## Plot results
fig, axs = plt.subplots(2,1, sharey=True)

bins = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2]

axs[0].hist(X[:, iters-1, 0], bins=bins)
axs[0].set_title("Distribuição iter {} para T = {}".format(iters, tempVec[0]))
axs[0].set_ylabel("Contagem")

axs[1].hist(X[:, iters-1, -1], bins=bins)
axs[1].set_title("Distribuição iter {} para T = {}".format(iters, tempVec[-1]))
axs[1].set_ylabel("Contagem")
axs[1].set_xlabel("Estado")

plt.suptitle("Histograma de X(t), N = {}".format(N), fontsize=14)

plt.show()
