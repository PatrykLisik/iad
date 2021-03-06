from .functions import random_point
from .functions import Euklides_dist as E_dist
import numpy as np
from collections import OrderedDict


class K_means():
    """Implementation of k-means"""

    def __init__(self, points_number, points_to_aprox, neighborhood_radius=2,
                 dist_func_points=E_dist, low=-10, high=10,
                 lazy_numer=0, lr=0.2,):
        """
        Args:
            points_number: number of points to approximate
            dinm_network: number of dimensions of network organization
            points_to_aprox: list of points to perform approximation on
            neighborhood_radius: one of parameter of net_dist_to_lr
            dist_func_points: function that return distnce between points
            low, higth: starting points are in square (low,low),(high,high)
        """

        self.neurons = self.genStartPos(points_number, low, high)
        self.points_to_aprox = points_to_aprox
        self.dist_func_points = dist_func_points
        self.dead_neurons = 0

    def genStartPos(self, n, low, high):
        """
        Params:
            n - number of centers
            low - minimum value of posion
            high - maximum value of posion
        Returns
            dictioanry of n random point to empy list
        """
        ans = {}
        for _ in range(n):
            ans[random_point(2, low, high)] = []
        return OrderedDict(ans)

    def match_points_to_neurons(self):
        neur = list(self.neurons.keys())
        for point in self.points_to_aprox:
            # find the closest neuron
            tcn = min(neur, key=lambda n: self.dist_func_points(n, point))
            # assign point to neuron
            self.neurons[tcn].append(point)

    def move_centers(self):
        """
        Deletes assign neurons position to mean of their points
        """

        dead_set = set([])
        for old_pos, points in self.neurons.items():
            # Don't do anything with tired neurons
            if(len(points) == 0):
                #print("BREAK:::", old_pos)
                dead_set.add(old_pos)
                continue
            # compute new posion
            x, y = zip(*points)
            avg_x = np.mean(x)
            avg_y = np.mean(y)
            # replace old neuron with new new without losing order
            self.neurons = OrderedDict([((avg_x, avg_y), []) if k == old_pos else (
                k, v) for k, v in self.neurons.items()])
        self.dead_neurons = min(len(dead_set), self.dead_neurons)

    def iter_once(self):
        self.match_points_to_neurons()
        self.move_centers()

    def getNeurons(self):
        ret = []
        for p in list(self.neurons.keys()):
            pp = list(p)
            ret.append(pp)
        return ret
