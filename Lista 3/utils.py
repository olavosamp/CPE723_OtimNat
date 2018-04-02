import numpy as np

def euclidean_dist(x, y):
    dist = x - y
    dist = np.power(dist, 2)
    return dist

def center_mass(x):
    return np.sum(x)/len(x)
