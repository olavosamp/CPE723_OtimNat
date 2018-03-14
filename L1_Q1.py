# Lista 1
# Questão 1

import random
import numpy as np

K = 1000

func = lambda x: x*np.exp(-x)   # Funcao: x exp(-x)

## Letra B

x = np.random.random(K)     # Distrib uniforme entre (0,1)

y = np.mean(func(x))

print("Letra B: ", y)
# print("Vetor X: ", x)

## Letra C

# Método de descartar valores de x maiores que 1
# e normalizar com o número total de valores testados
# Não funciona

# x = np.empty(K)
# K_real = 0
# n = 0
# while K_real < K:
#     x_new = np.random.exponential()
#     if x_new < 1:
#         x[K_real] = x_new
#         K_real += 1
#
#     n += 1
#
# y = np.sum(func(x))/n

# Método de normalizar a área da distrib para 1
# Funciona melhor, mas pior que a Letra B
x = np.random.exponential(size=K)/(1-np.exp(-1))
y = np.mean(func(x))

print("Letra C: ", y)

# print("Vetor X: ", x)
