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


def print_net(net):
    """
    Print network's neurons in console
    """
    for pos_net, pos_space in net.items():
        print("pos_net", pos_net)
        print("pos_space", pos_space)
    print()


def quantization_error(neurons, points, dist_func=Euklides_dist):
    """
    Args:
        Neurons: list of neuron posion in space
        points: list of point posion in space
        dist_func: callable object that compute distance in space
    Return:
        Mean of distances from every neuron to every point
    """
    err = 0
    for neuron in neurons:
        for point in points:
            err += dist_func(neuron, point)
    return err / len(neurons)


def quantization_error2(neurons, points, dist_func=Euklides_dist):
    """
    Args:
        Neurons: list of neuron posion in space
        points: list of point posion in space
        dist_func: callable object that compute distance in space
    Return:
        Mean of distances from every neuron to corresponding closest point
    """
    err = 0
    for neuron in neurons:
        # the closest point
        tcp = min(points, key=lambda p: dist_func(neuron, p))
        err += dist_func(neuron, tcp)
    return err / len(neurons)

    def random_point(n):
        """
        Generete random point in n-dimensional space
        Returns:
                n - element tuple of random 32-bit floats
        """
        # bounds are temporarily hardcoded to -10,10
        return tuple(np.random.uniform(low=-10, high=10, size=(n)))
