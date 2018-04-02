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
    numStates = len(J)
    transitionMatrix = np.zeros(numStates)

    for i in range(numStates):
        for j in range(numStates):
            if i != j:
                transitionMatrix[i][j] = (1/numStates)*np.exp(-(J[i] - J[j])/T)

    for j in range(numStates):
        transitionMatrix[j][j] = 1 - np.sum(transitionMatrix[:][j])

    return transitionMatrix
