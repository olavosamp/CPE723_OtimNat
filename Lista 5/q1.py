import numpy as np
import random

def roundRobin(Jpop,q=10,verbose=False):
    assert len(Jpop)>q,"Population size must be at larger than the number of oponents"
    scores = {"win":3,"loss":0,"draw":1}
    scoreTable = np.zeros(len(Jpop))
    for i in range(len(Jpop)):
        indexes = list(range(len(Jpop)))
        del indexes[i]
        if verbose:
            print("indexes=",indexes)
        random.shuffle(indexes)
        for j in range(q):
            #Select Oponent
            indexOponent = indexes[0]
            del indexes[0]
            if  Jpop[i]>Jpop[indexOponent]:
                scoreTable[i]           +=scores["win"]
                scoreTable[indexOponent]+=scores["loss"]
            elif Jpop[i]==Jpop[indexOponent]:
                scoreTable[i]           +=scores["draw"]
                scoreTable[indexOponent]+=scores["draw"]
            elif Jpop[i]<Jpop[indexOponent]:
                scoreTable[i]           +=scores["loss"]
                scoreTable[indexOponent]+=scores["win"]
            if verbose:
                print ("Selected: ",i," vs ",indexOponent)
                print ("Jpop[",i,"]=",Jpop[i]," vs ","Jpop[",indexOponent,"]=",Jpop[indexOponent])
                print ("scoreTable=",scoreTable)
        if verbose:
            print("\n")
    return scoreTable
