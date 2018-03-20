# Lista 1
# Questao 5

import numpy as np
import matplotlib.pyplot as plt

M = 1000
N = 4
T_0 = 0.1
e = 0.1
K = 10

J = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

x       = np.empty((M*K,N))
J_vec   = np.empty((M*K,1))

# x[0] = np.random.random()*10
x[0] = 5
print(x.shape)
T = T_0
k = 1
for i in range(0, M*K-1):
    r = np.random.random(N) - 0.5    # r pertence a (-0.5, 0.5)
    x_new = x[i] + e*r

    dJ = J(x_new) - J(x[i])

    prob = np.random.random(1)
    boltz = np.exp(-dJ/T)
    if dJ < 0:
        x[i+1] = x_new
        # print("New1")
    elif prob < boltz:
        x[i+1] = x_new
        # print("New2")
    else:
        x[i+1] = x[i]
        # print("Old")

    if np.mod(i, M) == 0:
        T = T_0/(np.log2(1 + k))
        k = k+1
    J_vec[i] = J(x[i])

print("x otimo: ", x[-1])


plt.plot(range(0, M*K), x)
plt.ylabel("x")
plt.xlabel("iter")
plt.show()


plt.plot(range(0, M*K), J_vec)
plt.ylabel("J(x)")
plt.xlabel("iter")
plt.show()
