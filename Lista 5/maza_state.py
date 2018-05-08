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

        val: 1 by 2 x ndims vector. First ndims elements are floating point state values, x_i.
            Last ndims elements are mutation rates, sigma_i.

            val = [x_0, x_1, ... x_i, sigma_1, ... , sigma_i]

    """
    ## Constructor
    def __init__(self, val=0.0, randomize=True, ndims=2, sigmaLimit=0.0001, initRange=2.0):
        self.sigmaLimit = sigmaLimit
        if ndims > 1:
            self.ndims = ndims
        else:
            raise ValueError("ndims must be greater than 1")

        if not(randomize):
            if len(val) == 2*self.ndims:
                self.setVal(val)
            else:
                raise ValueError("Value must agree with ndims")
        else:
            # If randomize is True, randomly initialize state
            self.randomState()

    # Set val attribute
    def setVal(self, val):
        if len(val) == 2*self.ndims:
            # Set value if it has correct dimensions
            self.val = val
        else:
            raise ValueError("Value must be a vector of lenght {}, but received length {}".format(2*self.ndims, len(val)))
        return self.val

    # Assign a random value for val attribute
    def randomState(self, initRange=2.0):
        # Initial value is a random number in the open interval ]-lim, +lim[
        self.val = np.random.rand(2*self.ndims)*initRange-initRange/2
        return self.val

    ## Compute state aptitude
    def aptitude(self):
        apt = ackley(self.val[0], self.val[1])
        return -apt

    ## Randomly alter state and sigma values
    def mutate_uncorr_multistep(self, tau1, tau2, mutateProb=1.0):
        '''
            Apply gaussian perturbation to state values as follows:
                x_i_new     = x_i + sigma_i * gauss1_i
                sigma_i_new = sigma_i * exp(tau1 * gaussNoise) * exp(tau2 * gauss2_i)

            Where:
                gauss_i are gaussian samples generated for each element
                gaussNoise is gaussian noise used for all elements

                tau1 = 1/sqrt(2*n)
                tau2 = 1/sqrt(2*sqrt(n))
                    Where n is population size
        '''
        # Evolutionary Strategy algorithms always perform mutation
        if mutateProb < 0:
            raise ValueError("Mutate probability must be a number between 0 and 1")
        if mutateProb > 1:
            mutateProb = 1.0

        randomNum = np.random.rand()
        if randomNum < mutateProb:
            # Sigma mutation
            gaussNoise = np.random.normal()
            sigmaNoise = np.random.normal(size=self.ndims)

            newSigma = self.val[:self.ndims]*np.exp(tau1*gaussNoise)*np.exp(tau2*sigmaNoise)

            # State mutation
            stateNoise = np.random.normal(size=self.ndims)
            newState = self.val[:self.ndims] + newSigma*stateNoise
            newVal = np.concatenate((newState, newSigma))

            self.setVal(newVal)

        return newVal

class mazaPop:
    '''
        Maza-Population class
    '''
    ## Constructor
    def __init__(self, size=50, seed=0, initRange=2.0):
        self.mutateProb = 1.0
        # self.numSurvivors = 50
        self.newPop     = []

        self.tau1 = 1/(np.sqrt(2*size))
        self.tau2 = 1/(np.sqrt(2*np.sqrt(size)))

        if seed != 0:
            np.random.seed(seed=seed)

        # Randomly initialize states
        self.size = size
        stateList = []
        for i in range(self.size):
            stateList.append(mazaState(randomize=True, initRange=initRange))

        aptList = np.zeros(self.size)
        self.popList = pd.DataFrame(data={"State":stateList, "Aptitude":aptList})
        # Compute aptitude list via corresponding function
        self.popList = compute_aptitudes(self.popList)

    ## Parent Selection
    def select_parents(self):
        '''
            Parent selection is not implemented for Evolutionary Programming algorithms
            Every specimen generates an offspring via mutation of its own genetic load
        '''
        return self.popList

    def survivor_selection_n_best(self):
        '''
            Select the specimens with highest aptitude, in a number equal to population size.
            Population list to be selected is the entire list of parents plus offspring.
        '''

        self.newPop  = compute_aptitudes(self.newPop)
        self.newPop  = self.newPop.sort_values("Aptitude", ascending=False)
        self.popList = self.newPop.iloc[:self.size, :].reset_index(drop=True)
        return self.popList

    def survivor_selection_tourney(self, numPlays=10):
        '''
            Tournament selection
            Every specimen competes against each other in numPlays (q) = 10 plays.
            In each play, the specimen with greater aptitude wins. After each play, the score is updated for each specimen.
                Win:    +1
                Draw:    0
                Lose:   -2

            After the Tournament ends, the highest scoring specimens are selected until population size is filled.
        '''
        winScore  = +1
        drawScore =  0
        loseScore = -1

        popSize = len(self.newPop)
        score = pd.DataFrame({"Score":np.zeros(popSize)})
        tourneyPop = pd.concat([self.newPop, score], axis=1)

        for i in range(popSize):
            opponents = np.random.randint(0, popSize, size=numPlays)

            currList = np.ones(numPlays)*tourneyPop.loc[i    , "Aptitude"]
            oppList  = tourneyPop.loc[opponents, "Aptitude"].values

            scoreList = np.zeros(numPlays)
            scoreList += np.where(currList > oppList, winScore, 0)
            scoreList += np.where(currList < oppList, loseScore, 0)

            tourneyPop.loc[i, "Score"] = np.sum(scoreList)

            # for index in opponents:
            #     if index != i:
            #         current  = tourneyPop.loc[i    , "Aptitude"]
            #         opponent = tourneyPop.loc[index, "Aptitude"]
            #         if current > opponent:
            #             tourneyPop.loc[i, "Score"] += winScore
            #         elif current < opponent:
            #             tourneyPop.loc[i, "Score"] += loseScore
            #         else:
            #             tourneyPop.loc[i, "Score"] += drawScore

        tourneyPop   = tourneyPop.sort_values("Score", ascending=False)
        self.popList = tourneyPop.iloc[:self.size, :2].reset_index(drop=True)
        # print(tourneyPop.head(10))
        # print(self.popList.head(10))

        return self.popList


    def mutation(self, population, mutateProb=1.0):
        for i in range(len(population)):
            population.loc[i, "State"].mutate_uncorr_multistep(self.tau1, self.tau2, mutateProb)

        self.newPop = population

        return self.newPop

    def generation(self):
        parents = self.select_parents()

        # self.recombination_n_bes(parents, numChildren=2)

        self.mutation(parents, mutateProb=self.mutateProb)
        self.newPop = pd.concat([self.newPop, self.popList]).reset_index(drop=True)

        self.survivor_selection_n_best()
        # self.survivor_selection_tourney()

        return self.popList

    #
    # def recombination(self, parents, numChildren=2):
    #     # Shuffle parents list
    #     parents = parents.sample(frac=1.0).reset_index(drop=True)
    #     parents = parents["State"]
    #     parentsLen = len(parents)
    #
    #     childList = []
    #     for i in range(int(self.numParents/2)):
    #         parentA = parents[i].val
    #         parentB = parents[parentsLen-i-1].val
    #         for j in range(numChildren):
    #             newChild = mazaState(val=crossover_1_point(parentA, parentB))
    #             childList.append(newChild)
    #
    #     # print(childList)
    #     self.newPop = pd.DataFrame(data={"State":childList, "Aptitude":np.zeros(len(childList))})
    #     self.newPop = compute_aptitudes(self.newPop)
    #     return self.newPop
