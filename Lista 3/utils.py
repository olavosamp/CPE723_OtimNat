import numpy as np

np.set_printoptions(precision=3)

def squared_dist(x, y):
    dist = x - y
    dist = np.power(dist, 2)
    dist = np.sum(dist)
    return dist

def dist_matrix(x, y):
    '''
    x: N by M matrix
    y: N by C matrix

    dist: M by C distance matrix
    '''



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

    # print(x[:, 0])
    # print(y[:,0]/T)
    # Compute P_y|x
    P = np.zeros((C, M))
    for l in range(C):
        for k in range(M):
            P[l, k] = (np.exp(-squared_dist(x[:,k], y[:,l])/T))  # P_(y_l)|(x_k)

    print("P shape: ", P.shape)
    w = np.sum(P, 0)       # Normalization factor w
    print("w shape: ", w.shape)
    # w = np.reshape(np.tile(w, M), (C, M))
    P = P/w    # Normalize each column

    return P

def cluster_cost(P, x, y):
    '''
    Compute cluster configuration cost, assuming equiprobable inputs.
    Cost is defined as:
        D = sum_x(x * sum_y(P_y|x))

    Arguments:
        P: C by M conditional probabilty matrix P_y|x
        x: N by M numpy array. N is input dimension. M is number of input elements
        y: N by C numpy array. C is number of centroids.
    Returns:
        cost: Scalar cost value D.
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
    # print(x)
    # print(y)

    # Compute cost
    cost = 0
    prob_x = 1/M    # Assuming equiprobable states
                    # If this is not true, use a list of probabilty weights
    for k in range(M):
        for l in range(C):
            current = P[l, k]*squared_dist(x[:, k], y[:, l])
            cost += current
            # print("dist(x_{}, y_{}) = dist(x_{}, y_{}): {}".format(k, l, x[:, k], y[:, l], current))

    cost *= prob_x
    cost = np.sum(cost)
    return cost

def centroid_update(P, x, C):
    '''
    Compute new centroid coordinates, according to:
        y_i = sum_x(x * P_yi|x)/sum_x(P_yi|x)

    Arguments:
        P: C by M conditional probabilty matrix P_y|x
        x: N by M numpy array. N is input dimension. M is number of input elements
        C: Number of centroids.
    Returns:
        yNew: N by C centroid coordinate matrix
    '''
    # Shenanigans to guarantee correct numpy dimensions
    if x.ndim == 1:
        M = len(x)
        N = 1
        x.shape = (1, M)
    else:
        N = np.shape(x)[0]
        M = np.shape(x)[1]

    yNew = np.zeros((N, C))
    for l in range(C):
        w = np.sum(P[l,:])           # Dividend sum_x(Pyi_x)
        yNew[:, l] = np.sum(x*P[l,:])/w

    # w = np.sum(P, 1)         # Divide each column by sum_x(Pyi_x)
    # print(w)
    # yNew = np.divide(yNew, w)   #

    return yNew
