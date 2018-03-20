import random
import math

x_k  = 9.869
# x_k1 =

dJ = lambda x, y: x**2 - y**2

R = random.random()*2 - 1
print("R: ", R)
r = random.random()
print("r: ", r)

x_k1 = x_k + 0.1*R
print("\nx_k1: ", x_k1)

q = lambda x: math.exp(-x/0.1)

dJ_now = dJ(x_k1, x_k)
print("\ndJ: ", dJ_now)
print("q: ", q(dJ_now))
