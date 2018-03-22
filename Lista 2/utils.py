def get_random_event(events, random_number):
    lower_bound = 0
    for i in range(len(events)):
        eventProb = events[i]
        # Check if random number is within state probabilty bound
        if lower_bound <= random_number <= lower_bound + eventProb:
            return i
        # If not, set bound for next state
        else:
            lower_bound += eventProb
