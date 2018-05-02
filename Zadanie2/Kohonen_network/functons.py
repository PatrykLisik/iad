import numpy as np
from scipy.spatial import distance


def Euklides_dist(x1, x2):
    """
    Euclidean distance:
    Square root of sum of squared substraction of every dimension
    """
    return distance.euclidean(x1, x2)


def RNF(R, distance):
    """
    rectangular neighborhood function
    R -  neighborhood radius
    distance - distance between dwo neurons in
    network topology

    Where R=0, this it winner takes all function
    """
    if distance <= R:
        return 1.0
    else:
        return 0.0


def GNF(R, distance):
    """
    Gaussian neighborhood function
    Args:
        R -  neighborhood radius
        distance - distance between dwo neurons in
        network topology
    """
    return np.exp(-(distance**2) / (2 * R**2))


def getPointsInCircle(center, radius, amount):
    centerX = center[0]
    centerY = center[1]

    ans = []
    while len(ans) < amount:
        randX = np.random.uniform(centerX - radius, centerX + radius)
        randY = np.random.uniform(centerY - radius, centerY + radius)
        if (centerX - randX)**2 + (centerY - randY)**2 < radius**2:
            ans.append([randX, randY])
    return ans


def print_net(net):
    """
    Print network's neurons in console
    """
    for pos_net, pos_space in net.items():
        print("pos_net", pos_net)
        print("pos_space", pos_space)
    print()
