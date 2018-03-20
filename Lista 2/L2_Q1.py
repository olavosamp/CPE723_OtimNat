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

print("\nProb 1:", random.random())
print("\nProb 2:", random.random())
print("\nProb 3:", random.random())
