import numpy                as np
# import matplotlib.pyplot    as plt

def flipBit(bit):
    if bit == 0:
        return 1
    if bit == 1:
        return 0

class mazaState:
    """
        docstring for mazaState.
    """
    def __init__(self, binVal=0, randomize=False):
        self.bitLen = 16
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

    def aptitude(self):
        x = self.intVal
        apt = x**2 - 0.3*np.cos(10*np.pi*x)
        return apt

    def randomState(self):
        return ''.join(np.random.randint(0, 2, size=16).astype('str'))

# state1 = mazaState('0001111111111100')
# randomString = ''.join(np.random.randint(0, 2, size=16).astype('str'))

state1 = mazaState('0001111111111100', randomize=True)
print("\nState X: ", state1.intVal)
print("J(X):      ", state1.aptitude())

# print(randomString)
# print(len(randomString))
