import numpy    as np
import pandas   as pd
import copy

# from tqdm import tqdm

# def flipBit(bit):
#     if bit == "0":
#         return "1"
#     if bit == "1":
#         return "0"
#     else:
#         raise ValueError("Argument must be a string value of 0 or 1")

def ackley(x):
    '''
    n-dimensional Ackley Cost Function

    x is the 1 by ndims vector of state values
    '''
    x = np.array(x)

    # Optionally test algorithm robustness by moving global optimum
    # x = x+1

    arg1 = -0.2*np.sqrt(np.mean(x*x))
    arg2 = np.mean(np.cos(2*np.pi*x))

    return -20.*np.exp(arg1) - np.exp(arg2) + 20. + np.e

def crossover_1_point(parentA, parentB, percentA=0.5):
    '''
    Single position crossover between parents A and B
    '''
    if len(parentA) != len(parentB):
        raise ValueError("Parent lengths must be equal")

    cut = int(round(len(parentA)*percentA))
    child = parentA[:cut] + parentB[cut:]
    return child

def copyPop(originalPop):
    size = len(originalPop)

    copyPop = originalPop.copy()
    for i in range(size):
        copyPop.loc[i, "State"] = copy.deepcopy(originalPop.loc[i, "State"])

    # print("Is same?", copyPop.loc[0, "State"] is originalPop.loc[0, "State"])
    return copyPop

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
            self.randomState(initRange)

    # Set val attribute
    def setVal(self, val):
        if len(val) == 2*self.ndims:
            # Set value if it has correct dimensions
            self.val = val
        else:
            raise ValueError("Value must be a vector of lenght {}, but received length {}".format(2*self.ndims, len(val)))
        # self.val = val
        return self.val

    # Assign a random value for val attribute
    def randomState(self, initRange=2.0):
        # Initial value is a random number in the open interval ]-lim, +lim[
        self.val = np.random.rand(2*self.ndims)*initRange-initRange/2
        return self.val

    ## Compute state aptitude
    def aptitude(self):
        apt = ackley(self.val[:self.ndims])
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

            newSigma = self.val[self.ndims:]*np.exp(tau1*gaussNoise)*np.exp(tau2*sigmaNoise)

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
    def __init__(self, size=50, seed=0, initRange=2.0, ndims=2):
        self.mutateProb = 1.0
        self.fitEvals   = 0
        # self.newPop     = []
        self.tauMod     = 0.9
        self.acceptPercent = 0.2

        # Compute sigma mutation parameters
        self.tau1 = 5*1/(np.sqrt(2*size))
        self.tau2 = 5*1/(np.sqrt(2*np.sqrt(size)))

        # Set fixed random generator seed, if required
        if seed != 0:
            np.random.seed(seed=seed)

        # Randomly initialize states
        self.size = size
        stateList = []
        for i in range(self.size):
            stateList.append(mazaState(randomize=True, initRange=initRange, ndims=ndims))

        aptList = np.zeros(self.size)
        self.popList = pd.DataFrame(data={"State":stateList, "Aptitude":aptList})

        # Compute aptitude list
        self.compute_aptitudes(self.popList)


    def compute_aptitudes(self, popList):
        # Compute fitness for every state
        popList["Aptitude"] = list(map(lambda x: x.aptitude(), popList["State"]))

        # Increment number of fitness evaluations
        self.fitEvals += len(popList)

        return popList


    ## Parent Selection
    def select_parents(self):
        '''
            Parent selection is not implemented for Evolutionary Programming algorithms
            Every specimen generates an offspring via mutation of its own genetic load
        '''
        return self.popList.copy(deep=True)


    def mutation(self, population, mutateProb=1.0):
        if self.acceptPercent > 0.2:
            self.tau1 /= self.tauMod
            self.tau2 /= self.tauMod

        elif self.acceptPercent < 0.2:
            self.tau1 *= self.tauMod
            self.tau2 *= self.tauMod

        # self.newPop = population.copy(deep=True)
        self.newPop = copyPop(population)
        #
        # print("\npopList--------------")
        # for state in self.popList["State"]:
        #     print(state.val)
        # print("--------------")
        # print(self.newPop["Aptitude"])
        # print(self.newPop.shape)
        # print("newPop--------------")
        # for state in self.newPop["State"]:
        #     print(state.val)
        # print("MAZA IS: ", self.popList.loc[0, "State"] is self.newPop.loc[0, "State"])

        for i in range(len(population)):
            self.newPop.loc[i, "State"].mutate_uncorr_multistep(self.tau1, self.tau2, mutateProb)

        # print("popList--------------")
        # for state in self.popList["State"]:
        #     print(state.val)
        # print("newPop--------------")
        # for state in self.newPop["State"]:
        #     print(state.val)

        # Update aptitudes
        # print(self.popList["Aptitude"])
        self.compute_aptitudes(self.newPop)

        self.acceptPercent = np.mean(self.newPop["Aptitude"] > self.popList["Aptitude"], axis=0)
        # print(self.newPop.shape)
        # print("newPop--------------")
        # # print(self.newPop["Aptitude"])
        # for state in self.newPop["State"]:
        #     print(state.val)

        # input()

        return self.newPop


    def survivor_selection_n_best(self):
        '''
            Select the specimens with highest aptitude, in a number equal to population size.
            Population list to be selected is the entire list of parents plus offspring.
        '''
        # print(self.newPop["Aptitude"])
        # for state in self.newPop["State"]:
        #     print(state.val)
        self.compute_aptitudes(self.newPop)
        # print("--------------")
        # for state in self.newPop["State"]:
        #     print(state.val)
        # print(self.newPop["Aptitude"])
        self.newPop.sort_values("Aptitude", ascending=False, inplace=True, kind='quicksort')


        self.popList = self.newPop.iloc[:self.size, :2].copy(deep=True).reset_index(drop=True)

        # print(self.popList["Aptitude"])

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
        loseScore = -2

        popSize    = len(self.newPop)
        score      = pd.DataFrame({"Score":np.zeros(popSize)})
        tourneyPop = pd.concat([self.newPop, score], axis=1)

        for i in range(popSize):
            opponents = np.random.randint(0, popSize, size=numPlays)

            currList = np.ones(numPlays)*tourneyPop.loc[i    , "Aptitude"]
            oppList  = tourneyPop.loc[opponents, "Aptitude"].values

            # Score changes of current contestant
            scoreList = np.zeros(numPlays)
            scoreList += np.where(currList > oppList,  winScore, 0)
            scoreList += np.where(currList < oppList,  loseScore, 0)
            # Uncomment for drawScore != 0
            # scoreList += np.where(currList == oppList, drawScore, 0)

            # Score changes of opponents
            # Not implemented, probably doesn't change the outcome

            tourneyPop.loc[i, "Score"] += np.sum(scoreList)

        tourneyPop.sort_values("Score", ascending=False, inplace=True)
        self.popList = tourneyPop.iloc[:self.size, :2].copy(deep=True).reset_index(drop=True)
        # print(tourneyPop.head(10))
        # print(self.popList.head(10))

        return self.popList


    def generation(self):
        parents = self.popList

        # self.recombination_n_best(parents, numChildren=2)

        self.mutation(parents, mutateProb=self.mutateProb)
        # self.compute_aptitudes(self.popList)
        self.compute_aptitudes(self.newPop)
        # AQUI ESTA O BUG

        self.newPop = pd.concat([self.popList, self.newPop], ignore_index=True)

        self.compute_aptitudes(self.newPop)

        self.survivor_selection_tourney()
        # self.survivor_selection_n_best()


        return self.popList
