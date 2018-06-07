import numpy as np
from scipy.spatial import distance
import math


def gaussRad(d, sig):
    return (1 / (np.sqrt(2.0 * math.pi) * sig)) * np.exp(-d**2 / (2 * sig**2))


def Euklides_dist(x1, x2):
    """
    Euclidean distance:
    Square root of sum of squared substraction of every dimension
    """
    return distance.euclidean(x1, x2)


def random_point(n, low=-10, high=10):
    """
    Generete random point in n-dimensional space
    Returns:
            n - element tuple of random 32-bit floats
    """
    return np.random.uniform(low=low, high=high, size=(n))
