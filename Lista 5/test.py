import copy
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
from tqdm                   import tqdm

from maza_state             import (mazaState,
                                    mazaPop)

numGenerations = 100
sizePop = 30
ndims = 30

# tau1 = 1/(np.sqrt(2*sizePop))
# tau2 = 1/(np.sqrt(2*np.sqrt(sizePop)))
#
# # state1 = mazaState(ndims=ndims)
# #
# # print(state1.val)
# # state1.mutate_uncorr_multistep(tau1, tau2)
# # print(state1.val)
#
# pop1 = mazaPop(size=sizePop)
# pop2 = copy.copy(pop1)
# # pop2 = pop1
# pop2.generation()
#
# print(pop1.popList["Aptitude"])
# print(pop2.popList["Aptitude"])
#
# # for i in range(len(pop2.popList)):
# #     print("Number ", i)
# #     print("pop1 val: {:.4f}".format(pop1.popList.loc[i, "Aptitude"]))
# #     # print("pop2 val: {:.4f}".format(pop2.popList.loc[i, "Aptitude"]))

df1 = pd.DataFrame({'maza': [1, 2, 3, 4], 'maza2': [5,6,7,8]})
df2 = pd.DataFrame({'maza': [0, 9, 8, 7], 'maza2': [6,5,4,3]})

print(df1)
print()
print(df2)

dfNovo = pd.concat([df1, df2], ignore_index=True)
print("")
print(dfNovo)
