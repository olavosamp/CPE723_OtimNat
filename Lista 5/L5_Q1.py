import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
from tqdm                   import tqdm

from maza_state             import (mazaState,
                                    mazaPop)

np.set_printoptions(precision=3)

numGenerations = 500
sizePop = 50
ndims = 3

numRuns = 10
bestFitOverall = []
fitEvals       = np.zeros(numRuns)
genCount       = np.zeros(numRuns)
for j in range(numRuns):
    pop1 = mazaPop(size=sizePop, initRange=60, ndims=ndims)

    popGen = []

    ## Run generations until convergence
    bestFit = [0, 1000]
    popMean = []
    tol      = 0.001
    patience = 0
    fitOld   = 1000
    while patience < 70:
        popGen.append(pop1.generation())
        fitNew = pop1.popList["Aptitude"].max()

        if np.abs(fitNew - fitOld) < tol:
            patience += 1
        else:
            patience = 0
            print("Best fit round {} Gen {}: {:2.4f}".format(j, len(popGen)+1, fitNew))

        fitOld = fitNew

        bestFit.append(popGen[-1]["Aptitude"].max())
        popMean.append(popGen[-1]["Aptitude"].mean())

    ## Run a specified number of generations
    # bestFit = [0, 1000]
    # popMean = []
    # for i in tqdm(range(numGenerations)):
    #     popGen.append(pop1.generation())
    #
    #     bestFit.append(popGen[i]["Aptitude"].max())
    #     popMean.append(popGen[i]["Aptitude"].mean())
    #     if  bestFit[i] > bestFit[i-1]:
    #         print("\nBest fit round {} Gen {}: {}".format(j, i+1, bestFit[i]))

    # Record number of generations and fitness evaluations
    genCount[j] = len(popGen)
    fitEvals[j] = pop1.fitEvals


    # for i in range(len(popGen)):
    #     # print("Best fit round {} Gen {}: {}".format(j, i+1, popGen[i]["Aptitude"].max()))
    #     # print("Mean fit round {} Gen {}: {}".format(j, i+1, popGen[i]["Aptitude"].mean()))
    #     bestFit.append(popGen[i]["Aptitude"].max())
    #     popMean.append(popGen[i]["Aptitude"].mean())

    # bestX = popGen[-1]["State"][29].val
    # print(bestX)
    bestFitOverall.extend(bestFit[2:-1])

    print("\n\nRun ", j)
    print("Run for {} generations".format(len(popGen)))
    print("Best Fitness: {:.3f}".format(bestFit[-1]))
    print("Mean Fitness: {:.3f}".format(popMean[-1]))

print("\nStatistics of {} runs".format(numRuns))
print("Best Fitness Mean: {}".format(np.mean(bestFitOverall)))
print("Best Fitness Std : {}".format(np.std(bestFitOverall)))
# print("Mean number of Generations until convergence         :\n{:.2f}".format(np.mean(genCount)))
# print("Mean number of fitness evaluations until convergence :\n{:.2f}".format(np.mean(fitEvals)))

plt.plot(bestFit[2:], 'r', label="Best Fitness")
plt.plot(popMean, 'b', label="Mean Fitness")
plt.legend()

plt.title("Fitness per Generation for run {}".format(numRuns))
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
