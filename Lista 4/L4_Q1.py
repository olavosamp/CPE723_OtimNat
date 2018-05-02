import numpy                as np
from maza_state             import mazaState, mazaPop, recombineAB
import matplotlib.pyplot    as plt

J = lambda x: x**2 - 0.3*np.cos(10*np.pi*x)

numGenerations = 50
pop1 = mazaPop(size=50)
popGen = []
for i in range(numGenerations):
    popGen.append(pop1.generation())


popMean = []
for i in range(len(popGen)):
    popMean.append(popGen[i]["Aptitude"].mean())

plt.plot(popMean, 'r')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
