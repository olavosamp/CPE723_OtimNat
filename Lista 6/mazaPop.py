import numpy    as np
import pandas   as pd
import copy


def compute_aptitude(states):
    aptitude = np.sum(states, axis=1)
    return aptitude

def parent_selection_roulette(population, numParents):
    popSize = population.shape[0]

    probabilities = population["Aptitude"]/population["Aptitude"].sum()

    parentsIndex = np.random.choice(np.arange(popSize), size=numParents, p=probabilities, replace=False)

    return parentsIndex

class MazaPop:
    '''
    pop: DataFrame size (size, 2). Each row is a specimen. Column "State" contains
            binary bitstrings. Column "Aptitude" contains corresponding fitness
            values for each specimen.
    '''
    def __init__(self, size=50, bitLen=20):
        self.size = size
        self.bitLen = bitLen

        # Initialize binary strings of length bitLen
        states = []
        for i in range(size):
            states.append(np.random.randint(0, high=2, size=self.bitLen))

        # Compute One-max fitness per specimen
        aptitude = compute_aptitude(states)

        self.pop = pd.DataFrame(data={"States":states, "Aptitude":aptitude})
