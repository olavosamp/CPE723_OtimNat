# Lista 1
# Questao 3

import numpy as np
import matplotlib.pyplot as plt

M = 10
N = 1
T = 0.1
e = 0.1

x = np.empty((M,N))
# x[0] = np.random.random()*50
x[0] = 10
print(x.shape)

for i in range(0, M-1):
    print("iter: ", i)
    r = np.random.random()*2 - 1    # r pertence a ()-1,1)
    print("R: ", r)
    x_new = x[i] + e*r

    J = np.power(x[i],2)
    J_new = np.power(x_new,2)

    dJ = J_new - J

    prob = np.random.random(1)
    boltz = np.exp(-dJ/T)
    print("R: {}\n x[n]: {}\nx[i+i] | {}".format(prob, boltz))
    if dJ < 0:
        x[i+1] = x_new
    elif prob < boltz:
        x[i+1] = x_new
    else:
        x[i+1] = x[i]

J = np.power(x,2)
plt.plot(x)
plt.ylabel("x")
plt.xlabel("iter")
plt.show()


plt.plot(J)
plt.ylabel("exp(-dJ/T)")
plt.xlabel("iter")
plt.show()
