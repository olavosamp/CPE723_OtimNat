import numpy    as np
import pandas   as pd

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
    print(len(child))
    return child

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
        return apt

    def mutate(self, mutateProb=0.02):
        if mutateProb < 0:
            raise ValueError("Mutate probability must be a number between 0 and 1")
        if mutateProb > 1:
            mutateProb = 1.0

        randomNum = np.random.rand()

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
        self.numParents = 10    # Number of parents selected. Must be pair
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
        self.popList = self.compute_aptitudes(self.popList)

    def compute_aptitudes(self, popList):
        aptList = list(map(lambda x: x.aptitude(), popList["State"]))
        popList["Aptitude"] = aptList
        return popList

    def select_parents(self):
        self.popList = self.popList.sort_values("Aptitude", ascending=False)
        parents = self.popList.iloc[:self.numParents, :]
        return parents

    def recombination(self, parents):
        # Shuffle parents list
        parents = parents.sample(frac=1.0)

        childList = []
        for i in range(int(self.numParents/2)):
            newChild = recombineAB(parents[i], parents[-i])
            childList.append(newChild)

        self.newPop = pd.DataFrame(data={"State":childList, "Aptitude":np.zeros(self.size)})
        self.newPop = pd.concatenate
        return self.newPop

    def survivor_selection(self):
        self.newPop  = self.compute_aptitudes(self.newPop)
        self.newPop  = self.newPop.sort_values("Aptitude", ascending=False)
        self.popList = self.newPop.iloc[:self.size, :]
        return self.popList
