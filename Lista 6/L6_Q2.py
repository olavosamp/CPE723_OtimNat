import numpy    as np
import pandas   as pd
import copy

from mazaPop    import (MazaPop,
                        parent_selection_roulette)

popSize = 20
bitLen  = 10
numParents = 10


'x = [x1, x2, ... xn], vector size: 1 by L'

mazaPop = MazaPop(size=popSize, bitLen=bitLen)

# print(mazaPop.pop.axes)
parentsIndex = parent_selection_roulette(mazaPop.pop, numParents)
print(mazaPop.pop.sort_values("Aptitude", ascending=False))
print("")
print(mazaPop.pop.loc[parentsIndex, :])
