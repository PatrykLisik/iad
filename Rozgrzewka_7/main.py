import numpy as np
from scipy.spatial import distance
import math


def Euklides_dist(x1, x2):
    """
    Euclidean distance:
    Square root of sum of squared substraction of every dimension
    """
    return distance.euclidean(x1, x2)


def random_point(n, min=0, max=10):
    """
    Generete random point in n-dimensional space
    Returns:
            n - element tuple of random 32-bit floats
    """
    return tuple(np.random.uniform(low=min, high=max, size=(n)))


def gaussRad(d, sig):
    return 1 / (np.sqrt(2 * math.pi) * sig) * np.exp(-d**2 / (2 * sig**2))


def genNeurons(n):
    neurons = {}
    center = random_point(2, 0, 10)
    weigth = random_point(2, -4, 4)


dick = {[i, j] for i in range(5) for j in range(5)}
