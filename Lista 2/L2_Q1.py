# Lista 2
# Questao 1
#
# CPE723
# Olavo Sampaio

import numpy                as np
import matplotlib.pyplot    as plt
import random

# Letra A

M = [[0.5, 0.25, 0.25],
     [0.25, 0.5, 0.25],
     [0.25, 0.25, 0.5]]

p0 = [[0.3],
      [0.4],
      [0.3]]

X3 = np.dot(np.linalg.matrix_power(M, 3),p0)

# print(X3)

# Letra B

# print("\nProb 1:", random.random())
# print("\nProb 2:", random.random())
# print("\nProb 3:", random.random())

# Letra C
# Compute X(t) for N  iterations

def get_random_event(events, random_number):
    lower_bound = 0
    for i in range(len(events)):
        eventProb = events[i]
        # Check if random number is within state probabilty bound
        if lower_bound <= random_number <= lower_bound + eventProb:
            return i
        # If not, set bound for next state
        else:
            lower_bound += eventProb

N = 100
t = 3

# Initialize X[0] with a state from 0 to 2
X = np.empty((N, t+1), dtype=int)
X[:,0] = np.random.randint(0, np.shape(M)[0],size=N)

randVec = np.random.random((N,t))
for i in range(N):
    for j in range(t):
        X[i,j+1] = get_random_event(M[X[i,j]][:], randVec[i,j])

# Letra D
# Create histograms of X(t)

bins = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2]

fig, axs = plt.subplots(2,2, sharey=True)

axs[0,0].hist(X[:,0], bins=bins)
axs[0,0].set_title("X(0)")
axs[0,0].set_ylabel("Contagem")

axs[0,1].hist(X[:,1], bins=bins)
axs[0,1].set_title("X(1)")

axs[1,0].hist(X[:,2], bins=bins)
axs[1,0].set_title("X(2)")
axs[1,0].set_xlabel("Estado")
axs[1,0].set_ylabel("Contagem")

axs[1,1].hist(X[:,3], bins=bins)
axs[1,1].set_title("X(3)")
axs[1,1].set_xlabel("Estado")

plt.suptitle("Histograma de X(t), N = {}".format(N), fontsize=14)
plt.show()
