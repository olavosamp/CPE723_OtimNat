import numpy                as np
# import matplotlib.pyplot    as plt

class mazaState:
    """
        docstring for mazaState.
    """
    def __init__(self, binVal=0):
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
        if len(binVal) == 16:
            self.binVal = binVal
            self.intVal = self.bin2int(self.binVal)
        else:
            raise ValueError("binVal string has wrong number of bits")
        return self.binVal


    def aptitude(self):
        x = self.intVal
        apt = x**2 - 0.3*np.cos(10*np.pi*x)
        return apt

# for i in range(-2, 3):
#     print("{:2d} | {}".format((i), np.binary_repr(i, width=4)))
#
# print("")
# for i in range(-2, 3):
#     # print("{:2d} | {}".format((i), int(np.binary_repr(i, width=4)), 2))
#     print("{:2d} | {}".format((i), int(bin(i), base=0)))

state1 = mazaState('0001111111111100')
# state1.setBin('0101111111111111')
print("\nState X: ", state1.intVal)
print("J(X):      ", state1.aptitude())
