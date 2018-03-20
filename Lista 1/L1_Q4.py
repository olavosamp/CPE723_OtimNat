# Lista 1
# Questao 3

import numpy as np
import matplotlib.pyplot as plt

M = 1000
N = 1
T_0 = 0.1
e = 0.1
K = 10

# J = lambda x: np.power(x,2)
J = lambda x: -x + 100*np.power(x - 0.2,2)*np.power(x - 0.8, 2)

x = np.empty((M*K,N))
# x[0] = np.random.random()*50
x[0] = 0
print(x.shape)
T = T_0
k = 1
for i in range(0, M*K-1):
    r = np.random.random() - 0.5    # r pertence a (-0.5, 0.5)
    x_new = x[i] + e*r
    # print(x_new)

    J_curr = J(x[i])
    J_new  = J(x_new)

    dJ = J_new - J_curr

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
        # print("T: ", T)
    # print("x: ", x[i])

print("x otimo: ", x[-1])


plt.plot(x)
plt.ylabel("x")
plt.xlabel("iter")
plt.show()

J_total = J(x)
plt.plot(J_total)
plt.ylabel("J(x)")
plt.xlabel("iter")
plt.show()
