import numpy as np

np.set_printoptions(precision=3)
#
def rms_error(x, y):
    dist = x - y
    dist = np.power(dist, 2)
    return dist

def center_mass(x):
    return np.sum(x)/len(x)

def prob_yx(x, y, T):
    '''
    Compute conditional probabilty matrix P_y|x
    Arguments:
        x: N by M numpy array. N is input dimension. M is number of input elements
        y: N by C numpy array. C is number of centroids.
        T: Boltzmann-Gibbs distribution temperature.
    Returns:
        P: C by M conditional probabilty matrix P_y|x
    '''
    # Shenanigans to guarantee correct numpy dimensions
    if x.ndim == 1:
        M = len(x)
        x.shape = (1, M)
    else:
        M = np.shape(x)[1]

    if y.ndim == 1:
        C = len(y)
        y.shape = (1, C)
    else:
        C = np.shape(y)[1]

    # Compute P_y|x
    P = np.zeros((C, M))
    for l in range(C):
        for k in range(M):
            P[l, k] = np.exp(-rms_error(x[:,k], y[:,l])/T)  # P_(y_l)|(x_k)
        P[l, :] = P[l, :]/np.sum(P[l, :])                   # Normalization factor u_x

    return P

def cluster_cost(P, x, y):
    '''
    Assuming equiprobable inputs
    '''

    # Shenanigans to guarantee correct numpy dimensions
    if x.ndim == 1:
        M = len(x)
        x.shape = (1, M)
    else:
        M = np.shape(x)[1]

    if y.ndim == 1:
        C = len(y)
        y.shape = (1, C)
    else:
        C = np.shape(y)[1]

    # Compute cost
    cost = 0
    prob_x = 1/M    # Assuming equiprobable states
                    # If this is not true, use a list of probabilty weights
    for k in range(M):
        for l in range(C):
            cost += P[l, k]*rms_error(x[:, k], y[:, l])

    cost /= prob_x
    return cost

def centroid_update(P, x, numCentroids):
    # Shenanigans to guarantee correct numpy dimensions
    if x.ndim == 1:
        M = len(x)
        N = 1
        x.shape = (1, M)
    else:
        N = np.shape(x)[0]
        M = np.shape(x)[1]

    yNew = np.zeros((N, numCentroids))
    for l in range(numCentroids):
        # print("\ny_{}:".format(l))
        for k in range(M):
            # print("\nx_{}: {}".format(k,x[:, k]))
            # print("P_{}|{}: {}".format(l,k,P[l, k]))
            yNew[:, l] = x[:, k]*P[l, k]
        yNew[:, l] /= np.sum(P[l, :])
        # print("Debug:", yNew)

    return yNew
