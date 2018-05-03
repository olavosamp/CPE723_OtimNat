import numpy    as np
import pandas   as pd
# from tqdm import tqdm

def flipBit(bit):
    if bit == "0":
        return "1"
    if bit == "1":
        return "0"
    else:
        raise ValueError("Argument must be a string value of 0 or 1")

def recombineAB(parentA, parentB, percentA=0.5):
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
    def __init__(self, binVal=1, randomize=False):
        self.bitLen = 16    # Fixed 16-bit length
        self.binVal = 0
        self.intVal = 0

        if randomize:
            binVal = self.randomState()
        self.setBin(binVal)

    def bin2int(self, binVal):
        bits    = len(binVal[2:])
        frac    = int(binVal[2:], base=2)/(2**bits -1)
        intPart = int(binVal[1])

        if int(binVal[0]) == 1:
            sign = -1
        else:
            sign = +1
        return sign*(intPart + frac)

    def setBin(self, binVal):
        if len(binVal) == self.bitLen:
            self.binVal = binVal
            self.intVal = self.bin2int(self.binVal)
        else:
            raise ValueError("binVal string has wrong number of bits:\n\tFound    {}\n\tExpected {}".format(len(binVal), self.bitLen))

        return self.binVal

    def randomState(self):
        randomInt = np.random.randint(0, 2, size=16)
        # print(randomInt)
        return ''.join(randomInt.astype('str'))

    def aptitude(self):
        x = self.intVal
        # print(x)
        apt = x**2 - 0.3*np.cos(10*np.pi*x)
        return -apt

    def mutate(self, mutateProb=0.02):
        if mutateProb < 0:
            raise ValueError("Mutate probability must be a number between 0 and 1")
        if mutateProb > 1:
            mutateProb = 1.0

        randomNum = np.random.rand()
        newBinVal = 0
        if randomNum < mutateProb:
            mutatePos = np.random.randint(0, self.bitLen-1)
            newBinVal = self.binVal[:mutatePos] + flipBit(self.binVal[mutatePos]) + self.binVal[mutatePos+1:]
            self.setBin(newBinVal)

        return newBinVal

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
            parentA = parents[i].binVal
            parentB = parents[parentsLen-i-1].binVal
            for j in range(numChildren):
                newChild = mazaState(binVal=recombineAB(parentA, parentB))
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
