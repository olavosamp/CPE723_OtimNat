import pandas   as pd
import numpy    as np
from copy       import copy

def update_aptitude(pop, lamarck=False):
    # Recalculate One-max fitness per specimen
    aptitude = list(map(lambda x: np.sum(x), pop["State"]))
    pop["Aptitude"] = aptitude

    ## Update Memetic Aptitudes
    newStates, memeticAptitudes = compute_memetic_aptitudes(pop)

    if lamarck:
        for i in range(newStates.shape[0]):
            pop.loc[i, "State"][:] = newStates[i, :]

    pop["Memetic"] = memeticAptitudes
    return pop

def bit_flip(bit):
    if bit == 0:
        return 1
    if bit == 1:
        return 0
    else:
        raise ValueError("Argument must be a integer value of 0 or 1")

def compute_aptitudes(state):
    aptitude = np.sum(state, axis=1)
    return aptitude

def greedy_local_search(state):
    bitLen = len(state)
    apt = np.sum(state)

    position = 0
    newApt = 0
    # print(apt)
    while (newApt < apt) and (position < bitLen):
        newState = copy(state)
        newState[position] = bit_flip(newState[position])

        newApt = np.sum(newState)
        position += 1

    return newState, newApt

def compute_memetic_aptitudes(pop):
    popSize = pop.shape[0]
    bitLen  = len(pop.loc[0, "State"])

    # Local search
    newStates           = np.zeros((popSize, bitLen), dtype=int)
    memeticAptitudes    = np.zeros(popSize)
    for i in range(popSize):
        newStates[i,:], memeticAptitudes[i] = greedy_local_search(pop.loc[i, "State"])

    return newStates, memeticAptitudes

'Parent Selection'
def parent_selection_roulette(population, numParents):
    '''
    Input: population DataFrame
    '''
    popSize = population.shape[0]

    # probabilities = population["Aptitude"]/population["Aptitude"].sum()
    probabilities = population["Memetic"]/population["Memetic"].sum()

    parentsIndex = np.random.choice(range(popSize), size=numParents, p=probabilities, replace=True)

    return parentsIndex

'One-Point Crossover'
def crossover_one_point(parentA, parentB, prob=0.9):
    '''
    Input:  A pair of 1-row parent DataFrames
    Output: A pair of 1-row children DataFrames
    '''
    randomNum = np.random.rand()

    if randomNum < prob:
        childA, childB = recombineAB(parentA["State"], parentB["State"])

        # childA = parentA.copy()
        # childA["State"] = stateA
        #
        # childB = parentB.copy()
        # childB["State"] = stateB
    else:
        childA = parentA.loc[:, "State"]
        childB = parentB.loc[:, "State"]
    return childA, childB

'Recombination routine'
def recombineAB(parentA, parentB, percentA=0.5):
    '''
    Input:  A pair of parent state arrays
    Output: A pair of children state arrays
    '''
    if len(parentA) != len(parentB):
        raise ValueError("Parent lengths must be equal")

    childA = np.zeros(len(parentA), dtype=int)
    childB = np.zeros(len(parentA), dtype=int)
    cut = int(round(len(parentA)*percentA))

    childB[cut:] = parentB[cut:].copy()
    childB[:cut] = parentA[:cut].copy()

    childA[cut:] = parentA[cut:].copy()
    childA[:cut] = parentB[:cut].copy()

    return childA, childB

def mutation(specimen, prob=0.05):
    randomNum = np.random.rand()
    if randomNum < prob:
        mutatePos = np.random.randint(len(specimen))
        specimen[mutatePos] = bit_flip(specimen[mutatePos])

    return specimen

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
        state = []
        for i in range(size):
            state.append(np.random.randint(0, high=2, size=self.bitLen))

        # Compute One-max fitness per specimen
        aptitude = compute_aptitudes(state)

        self.pop = pd.DataFrame(data={"State":state, "Aptitude":aptitude})

    def update_aptitude(self, lamarck=False):
        # Recalculate One-max fitness per specimen
        aptitude = compute_aptitudes(state)
        self.pop["Aptitude"] = aptitude

        ## Update Memetic Aptitudes
        newStates, memeticAptitudes = compute_memetic_aptitudes(self.pop)

        if lamarck:
            for i in range(newStates.shape[0]):
                self.pop.loc[i, "State"][:] = newStates[i, :]

        self.pop["Memetic"] = memeticAptitudes

        return aptitude

def generation(pop, crossProb=0.9, mutateProb=0.05, lamarck=False):
    numParents = pop.shape[0]

    # Update memetic aptitudes
    pop = update_aptitude(pop, lamarck=lamarck).copy()

    zeroPad = np.zeros(pop.shape[0])
    newPop = pd.DataFrame(data={"State":zeroPad, "Aptitude":zeroPad})

    ## Parent selection
    parentsIndex = parent_selection_roulette(pop, numParents)
    parents = pop.loc[parentsIndex, :].reset_index(drop=True)

    ## Recombination
    # Each parent will create one child
    childrenStates = []
    for i in range(0, numParents, 2):
        childA, childB = crossover_one_point(parents.loc[i, :], parents.loc[i+1, :], prob=1.0)
        childrenStates.append(childA)
        childrenStates.append(childB)

    newPop["State"] = childrenStates

    # Update aptitudes
    newPop["Aptitude"] = list(map(lambda x: np.sum(x), newPop["State"]))

    ## Mutation
    newPop["State"] = list(map(mutation, newPop["State"]))
    newPop["Aptitude"] = list(map(lambda x: np.sum(x), newPop["State"]))

    ## Update Memetic Aptitudes
    newPop = update_aptitude(newPop, lamarck=lamarck).copy()

    return newPop
