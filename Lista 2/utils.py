import numpy as np

# Choose an event from a probabilty weighted list
def get_random_event(events, random_number):
    lower_bound = 0
    for i in range(len(events)):
        eventProb = events[i]
        # Check if random number is within state probabilty bound
        if lower_bound <= random_number <= lower_bound + eventProb:
            return i    # Return selected event
        # If not, set bound for next state
        else:
            lower_bound += eventProb

# Create a perturbation in a state vector X with N possible states
def perturb(x, states):
    randPos = np.random.randint(0, len(x))
    randVal = np.random.randint(0, states)
    # print(x.shape)
    # print(randVec.shape)
    xNew = x
    xNew[randPos] = randVal
    return xNew

def transition_matrix(J, T):
    '''
    Compute transition matrix for the discrete Metropolis algorithm of a function J(x).
    Arguments:
        J: Size N vector of cost values for N unique states. Each element should correspond to J(x) at the given state.
        T: temperature at which the matrix is to be calculated.
    Returns:
        M: NxN square Transition matrix for temperature T.
    '''
    numStates = len(J)
    transitionMatrix = np.zeros(numStates)

    for i in range(numStates):
        for j in range(numStates):
            if i != j:
                transitionMatrix[i][j] = np.exp(-(J[i] - J[j])/T)
                # transitionMatrix[i][j] *= (1/numStates) # Generator probabilty

    for j in range(numStates):
        transitionMatrix[j][j] = 1 - np.sum(transitionMatrix[:][j])

    return transitionMatrix

def metropolis_prob(J_Old, J_New, T):
    '''
    Compute state acceptance probabilty, according to Metropolis algorithm
    Args:
        J_Old:  Current state energy/cost
        J_New:  Candidate state energy
        T:      Temperature
    Returns:
        prob: acceptance probabilty
    '''
    diff = J_New - J_Old
    if diff < 0:
        prob = 1
    else:
        prob = np.exp(-(diff)/T)

    return prob
