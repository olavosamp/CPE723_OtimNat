
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import copy
# from tqdm                   import tqdm

from mazaPop    import (MazaPop,
                        parent_selection_roulette,
                        crossover_one_point,
                        mutation,
                        generation)

popSize = 100
bitLen  = 25
mutateProb = 1/bitLen
crossProb = 0.7
numGenerations = 100
lamarck = False

'x = [x1, x2, ... xn], vector size: 1 by L'

mazaPop = MazaPop(size=popSize, bitLen=bitLen)

popGen = []
for i in range(numGenerations):
    mazaPop.pop = generation(mazaPop.pop, crossProb=crossProb,
                             mutateProb=mutateProb, lamarck=lamarck).copy(deep=True)
    popGen.append(mazaPop.pop)

# Evaluation and plots
bestFit = []
popMean = []
bestFitMeme= []
popMeanMeme= []
for i in range(len(popGen)):
    bestFit.append(popGen[i]["Aptitude"].max())
    popMean.append(popGen[i]["Aptitude"].mean())
    bestFitMeme.append(popGen[i]["Memetic"].max())
    popMeanMeme.append(popGen[i]["Memetic"].mean())

print("\nBest Fitness: {:.2f}".format(bestFit[-1]))
print("Mean Fitness: {:.2f}".format(popMean[-1]))

plt.plot(bestFit, 'r', label="Best Fitness")
plt.plot(popMean, 'b', label="Mean Fitness")
plt.plot(bestFitMeme, 'g', label="Best Memetic Fitness")
plt.plot(popMeanMeme, 'k', label="Mean Memetic Fitness")
plt.legend()

plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.show()
