from .functions import random_point
from .functions import Euklides_dist as E_dist
import numpy as np


class K_menas():
    """Implementation of k-means"""

    def __init__(self, points_number, points_to_aprox, neighborhood_radius=2,
                 dist_func_points=E_dist,
                 lazy_numer=0, lr=0.2):
        """
        Args:
            points_number: number of points to approximate
            dinm_network: number of dimensions of network organization
            dist_func: callable object that takes two points of from
                        points_to_aprox and returns distance between them
            points_to_aprox: list of points to perform approximation on
            neighborhood_radius: one of parameter of net_dist_to_lr
            dist_func_points: function that return distnce between points
        """

        self.neurons = self.genStartPos(points_number)
        self.points_to_aprox = points_to_aprox
        self.dist_func_points = dist_func_points
        self.dead_neurons = 0

    def genStartPos(self, n):
        """
        Params:
            n - number of centers
        Returns
            dictioanry of n random point to empy list
        """
        ans = {}
        for _ in range(n):
            ans[random_point(2)] = []
        return ans

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
                print("BREAK:::", old_pos)
                dead_set.add(old_pos)
                continue
            # compute new posion
            x, y = zip(*points)
            avg_x = np.mean(x)
            avg_y = np.mean(y)
            # delete old neuron
            self.neurons.pop(old_pos)
            # add new
            self.neurons[(avg_x, avg_y)] = []
        self.dead_neurons = min(len(dead_set), self.dead_neurons)

    def iter_once(self):
        self.match_points_to_neurons()
        self.move_centers()

    def getNeurons(self):
        return self.neurons.keys()
