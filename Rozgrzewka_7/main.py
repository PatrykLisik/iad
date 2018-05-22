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


number_of_centers = 4
weigths = np.random.uniform(low=-4, high=4, size=(number_of_centers))
centers = np.random.uniform(low=0, high=10, size=(number_of_centers))

weigths_iter = iter(weigths)
centers_iter = iter(centers)
w0 = np.random.uniform(low=-4, high=4)

plt.grid()
x_all = np.linspace(0, 10, 1000)
sig = 0.2
y_all = [0 for _ in range(len(x_all))]
for w, c in zip(weigths, centers):
    y = []
    for x in x_all:
        d = Euklides_dist(x, c)
        y.append(w0 + w * gaussRad(d, sig))
    y_all = list(map(operator.add, y_all, y))
    # plot single center
    plt.plot(x_all, y, color="red")
# plot sum of all centers
plt.plot(x_all, y_all, color="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("out1")
plt.show()
