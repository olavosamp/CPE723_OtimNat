# Lista 1
# Questão 2

import random
import numpy as np

K = 1000

func = lambda x: np.sqrt(1 - x**2)   # Funcao: Área do semi círculo
                                     # y = sqrt(1 - x²)

## Letra B

x = np.random.random(K)     # Distrib uniforme entre (0,1)

y = np.mean(func(x))

print("Letra B: ", y)
print("4y = pi: ", y*4)
