# Lista 2
# Questao 1
import numpy as np
import random

# Letra A

M = [[0.5, 0.25, 0.25],
     [0.25, 0.5, 0.25],
     [0.25, 0.25, 0.5]]

p0 = [[0.3],
      [0.4],
      [0.3]]

X3 = np.dot(np.linalg.matrix_power(M, 3),p0)

print(X3)

# Letra B

def get_random_event(events):
    random_number = np.random.random()
    lower_bound = 0
    for eventProb in events:
        if lower_bound <= random_number <= lower_bound + eventProb:
            return events.index(eventProb)
        else:
            lower_bound += eventProb

# print("\nProb 1:", random.random())
# print("\nProb 2:", random.random())
# print("\nProb 3:", random.random())

# Letra C

N = 100
t = 3

X = np.empty((N, t+1), dtype=int)
X[:,0] = np.random.randint(0, np.shape(M)[0],size=N)
# print(X[:,0].shape)
print(X.shape)

for i in range(N):
    for j in range(t):
        # print(i,j)
        # print((X[i,j]))
        X[i,j+1] = get_random_event(M[X[i,j]])
print(X)
