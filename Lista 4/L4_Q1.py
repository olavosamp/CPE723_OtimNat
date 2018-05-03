import numpy                as np
from maza_state             import mazaState, mazaPop, recombineAB
import matplotlib.pyplot    as plt
from tqdm                   import tqdm

J = lambda x: x**2 - 0.3*np.cos(10*np.pi*x)

numGenerations = 200
pop1 = mazaPop(size=100)
popGen = []
for i in tqdm(range(numGenerations)):
    popGen.append(pop1.generation())


bestFit = []
popMean = []
for i in range(len(popGen)):
    bestFit.append(popGen[i]["Aptitude"].max())
    popMean.append(popGen[i]["Aptitude"].mean())

print("\nBest Fitness: {:.3f}".format(bestFit[-1]))
print("Mean Fitness: {:.3f}".format(popMean[-1]))

plt.plot(bestFit, 'r', label="Best Fitness")
plt.plot(popMean, 'b', label="Mean Fitness")
plt.legend()

# plt.ylim(ymin=-1.0)

plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.show()
