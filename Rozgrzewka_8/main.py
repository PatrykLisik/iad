import numpy as np
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
import operator


def Euklides_dist(x1, x2):
    """
    Euclidean distance:
    Square root of sum of squared substraction of every dimension
    """
    return distance.euclidean(x1, x2)


def gaussRad(d, sig):
    return (1 / (np.sqrt(2 * math.pi) * sig)) * np.exp(-d**2 / (2 * sig**2))


def gaussRad_d(d, sig):
    return gaussRad(d, sig) * (-1 * d / sig**2)


def RBF(centers_weigths, sig, wo, x):
    ret = w0
    for w, c in centers_weigths.items():
        d = Euklides_dist(x, c)
        ret += w * gaussRad(d, sig)
    return ret


def w_grad(weigth, center, sig, xs, ys):
    """
    Params:
        weigth,center - neuron params
        sig - sigma, neighborhood_radius
        xs - array of x
        ys - array of target value
    """


number_of_centers = 4
weigths = np.random.uniform(low=-4, high=4, size=(number_of_centers))
centers = np.random.uniform(low=0, high=10, size=(number_of_centers))

weigths_iter = iter(weigths)
centers_iter = iter(centers)
w0 = np.random.uniform(low=-4, high=4)

centers_weigths = dict(zip(centers, weigths))

x_all = np.linspace(0, 10, 1000)
sig = 0.2


def RBF_2(x): return RBF(centers_weigths, sig, w0, x)


y_all = []
for x in x_all:
    y_all.append(RBF_2(x))
plt.plot(x_all, y_all)
plt.show()
