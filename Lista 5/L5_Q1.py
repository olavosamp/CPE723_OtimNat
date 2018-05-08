import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
from tqdm                   import tqdm

from maza_state import (mazaState,
                        mazaPop)

numGenerations = 25
sizePop = 30

numRuns = 1
bestFitOverall = []
for j in range(numRuns):
    pop1 = mazaPop(size=sizePop, initRange=60)

    popGen = []
    for i in tqdm(range(numGenerations)):
        popGen.append(pop1.generation())

    bestFit = []
    popMean = []
    for i in range(len(popGen)):
        bestFit.append(popGen[i]["Aptitude"].max())
        popMean.append(popGen[i]["Aptitude"].mean())

    bestFitOverall.append(bestFit[-1])

    print("\n\nRun ", j)
    print("Best Fitness: {:.3f}".format(bestFit[-1]))
    print("Mean Fitness: {:.3f}".format(popMean[-1]))

print("\nStatistics of {} runs".format(numRuns))
print("Best Fitness Mean: {:.3f}".format(np.mean(bestFitOverall)))
print("Best Fitness Std : {:.3f}".format(np.std(bestFitOverall)))

plt.plot(bestFit, 'r', label="Best Fitness")
plt.plot(popMean, 'b', label="Mean Fitness")
plt.legend()

plt.title("Fitness per Generation for one run")
plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.show()

# tau1 = 1/(np.sqrt(2*sizePop))
# tau2 = 1/(np.sqrt(2*np.sqrt(sizePop)))
#
# state1 = mazaState()
#
# print(state1.val)
# print(state1.mutate_uncorr_multistep(tau1, tau2))
# pop1 = mazaPop(size=sizePop, seed=42)
# pop2 = mazaPop(size=sizePop, seed=42)
# pop2.generation()
#
# for i in range(len(pop2.popList)):
#     print("Number ", i)
#     print("pop1 val: {:.4f}".format(pop1.popList.loc[i, "Aptitude"]))
#     print("pop2 val: {:.4f}".format(pop2.popList.loc[i, "Aptitude"]))
