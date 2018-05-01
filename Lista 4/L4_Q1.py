import numpy                as np
from maza_state             import mazaState, mazaPop, recombineAB




# state1 = mazaState('0001111111111100')
# randomString = ''.join(np.random.randint(0, 2, size=16).astype('str'))

# state1 = mazaState('0001111111111100', randomize=True)
# for i in range(500):
#     state1 = mazaState(randomize=True)
    # print("\nState X: ", state1.intVal)
    # print("J(X):      ", state1.aptitude())
# print("Ok")

# pop1 = mazaPop(size=20)

# # print(pop1.stateList)
# pop1.selectParents()
# print(pop1.stateList)
# print(pop1.selectParents())
# # print(pop1.stateList.loc[0, "State"].aptitude())
# # print(pop1.stateList["Aptitude"])
# # pop1.stateList.
# # print(pop1.stateList["Aptitude"])

state1 = mazaState(randomize=True)
print(state1.binVal)

state2 = mazaState(randomize=True)
print(state2.binVal)

print(recombineAB(state1.binVal, state2.binVal))
# print(state1.aptitude())
# state1.mutate(mutateProb=1.0)
# print(state1.binVal)
# print(state1.aptitude())
