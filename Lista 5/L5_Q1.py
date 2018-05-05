import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
# from tqdm                   import tqdm

from maza_state import (mazaState,
                        mazaPop)

numGenerations = 10
sizePop = 50

tau1 = 1/(np.sqrt(2*sizePop))
tau2 = 1/(np.sqrt(2*np.sqrt(sizePop)))

state1 = mazaState()

print(state1.val)
print(state1.mutate(tau1, tau2))
# pop1 = mazaPop(size=sizePop)
# popGen = []
# for i in tqdm(range(numGenerations)):
#     popGen.append(pop1.generation())
#
# bestFit = []
# popMean = []
# for i in range(len(popGen)):
#     bestFit.append(popGen[i]["Aptitude"].max())
#     popMean.append(popGen[i]["Aptitude"].mean())
#
# print("\nBest Fitness: {:.3f}".format(bestFit[-1]))
# print("Mean Fitness: {:.3f}".format(popMean[-1]))
#
# plt.plot(bestFit, 'r', label="Best Fitness")
# plt.plot(popMean, 'b', label="Mean Fitness")
# plt.legend()
#
# # plt.ylim(ymin=-1.0)
#
# plt.xlabel("Generation")
# plt.ylabel("Fitness")
#
# plt.show()