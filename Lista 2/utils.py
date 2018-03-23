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

    randVec = np.random.randint(0, states, size=np.shape(x))
    print(x.shape)
    print(randVec.shape)
    return x + randVec
