import numpy    as np
import pandas   as pd
# from tqdm import tqdm

# def flipBit(bit):
#     if bit == "0":
#         return "1"
#     if bit == "1":
#         return "0"
#     else:
#         raise ValueError("Argument must be a string value of 0 or 1")

def ackley(x, y):
    '''
    Benchmark Ackley Function
    '''
    arg1 = -0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x) + np.cos(2. * np.pi * y))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

def crossover_1_point(parentA, parentB, percentA=0.5):
    '''
    Single position crossover between parents A and B
    '''
    if len(parentA) != len(parentB):
        raise ValueError("Parent lengths must be equal")

    cut = int(round(len(parentA)*percentA))
    child = parentA[:cut] + parentB[cut:]
    return child

def compute_aptitudes(popList):
    aptList = list(map(lambda x: x.aptitude(), popList["State"]))
    popList["Aptitude"] = aptList
    return popList

class mazaState:
    """
        maza-State/specimen class
    """
    def __init__(self, val=0.0, randomize=True, self.ndims=2):
        if ndims > 0:
            self.ndims = ndims
        else:
            raise ValueError("ndims must be greater than 0")

        if not(randomize):
            if len(val) == 2*self.ndims:
                self.setVal(val)
            else:

        else:
            # If randomize is True, randomly initialize state
            self.randomState(ndims=self.ndims)

    # def bin2int(self, binVal):
    #     bits    = len(binVal[2:])
    #     frac    = int(binVal[2:], base=2)/(2**bits -1)
    #     intPart = int(binVal[1])
    #
    #     if int(binVal[0]) == 1:
    #         sign = -1
    #     else:
    #         sign = +1
    #     return sign*(intPart + frac)

    def setVal(self, val):
        if len(val) == 2*self.ndims:
            # Set value if it has correct dimensions
            self.val = val
        else:
            raise ValueError("Value must be a vector of lenght {}, but received length {}".format(self.ndims, len(val)))
        return self.val

    def randomState(self):
        # Initial value is a random number in the interval ]-1, +1[
        self.val = np.random.rand(2*self.ndims)*2-1
        return self.val

    def aptitude(self):
        apt = ackley(self.val[0], self.val[1])
        return -apt

    def mutate(self, mutateProb=0.02, alpha=0.01):
        if mutateProb < 0:
            raise ValueError("Mutate probability must be a number between 0 and 1")
        if mutateProb > 1:
            mutateProb = 1.0

        randomNum = np.random.rand()
        if randomNum < mutateProb:
            mutateNoise =
            newval =
            self.setVal(newval)

        return newval

class mazaPop:
    '''
        Maza-Population class
    '''
    def __init__(self, size=50):
        self.numParents = 20    # Number of parents selected. Must be pair
        self.mutateProb = 0.01
        self.newPop     = []
        self.size       = 50

        # Randomly initialize states
        self.size = size
        stateList = []
        for i in range(self.size):
            stateList.append(mazaState(randomize=True))

        aptList = np.zeros(self.size)
        self.popList = pd.DataFrame(data={"State":stateList, "Aptitude":aptList})
        self.popList = compute_aptitudes(self.popList)


    def select_parents(self):
        aptSum = self.popList["Aptitude"].sum()
        probList = pd.DataFrame({"Probabilities": self.popList["Aptitude"].div(aptSum)})
        candidateList = pd.concat([self.popList, probList], axis=1)

        popSize = len(candidateList)
        parents = []
        while (len(parents) < self.numParents):
            for i in range(popSize):
                randomNum = np.random.rand()
                if (candidateList.loc[i, "Probabilities"] > randomNum) and (len(parents) < self.numParents):
                    parents.append(candidateList.loc[i, :])

        return pd.DataFrame(parents)

    def recombination(self, parents, numChildren=2):
        # Shuffle parents list
        parents = parents.sample(frac=1.0).reset_index(drop=True)
        parents = parents["State"]
        parentsLen = len(parents)

        childList = []
        for i in range(int(self.numParents/2)):
            parentA = parents[i].val
            parentB = parents[parentsLen-i-1].val
            for j in range(numChildren):
                newChild = mazaState(val=crossover_1_point(parentA, parentB))
                childList.append(newChild)

        # print(childList)
        self.newPop = pd.DataFrame(data={"State":childList, "Aptitude":np.zeros(len(childList))})
        self.newPop = compute_aptitudes(self.newPop)
        return self.newPop

    def survivor_selection(self):
        self.newPop  = compute_aptitudes(self.newPop)
        self.newPop  = self.newPop.sort_values("Aptitude", ascending=False)
        self.popList = self.newPop.iloc[:self.size, :].reset_index(drop=True)
        return self.popList

    def mutation(self, mutateProb=0.01):
        for i in range(len(self.newPop)):
            self.newPop.loc[i, "State"].mutate(mutateProb)
        return self.newPop

    def generation(self):
        parents = self.select_parents()
        self.recombination(parents, numChildren=2)
        self.newPop = pd.concat([self.newPop, self.popList]).reset_index(drop=True)
        self.mutation(mutateProb=0.01)
        self.survivor_selection()

        return self.popList
