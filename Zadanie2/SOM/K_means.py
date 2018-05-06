from .functions import random_point
from .functions import Euklides_dist as E_dist
from .functions import GNF
import numpy as np


class K_menas():
    """Implementation of k-means"""

    def __init__(self, points_number, points_to_aprox, neighborhood_radius=2,
                 net_dist_to_lr=GNF, dist_func_points=E_dist,
                 lazy_numer=None, lr=0.2):
        """
        Args:
            points_number: number of points to approximate
            dinm_network: number of dimmensions of network organization
            dist_func: callable objest that takes two points of from
                        points_to_aprox and returns distance beetwen them
            net_dist_to_lr: callable object that takes posion of two neurons
                            and returns leraning_rate_multiplayer. Throu this
                            argument WTA nad WTM approach can be achived
            points_to_aprox: list of points to perform approximation on
        """
        # self.neighborhood_radius = int(points_number / 5) + 1
        if(lazy_numer is None):
            self.lazy_numer = int(np.sqrt(points_number))
        else:
            self.lazy_numer = lazy_numer
        self.lr = lr
        self.neurons = self.genStartPos(points_number)
        self.points_to_aprox = points_to_aprox
        self.dist_func_points = dist_func_points

    def genStartPos(self, n):
        """
        Params:
            n - number of centers
        Returns
            dictioanry of n random point to empy list
        """
        ans = {}
        for _ in range(n):
            ans[random_point()] = []
        return ans

    def match_points_to_neurons(self):
        neurons = self.neurons.keys()
        for point in self.points_to_aprox:
            # find the closest neuron
            tcn = min(neurons, key=lambda n: self.dist_func_points(n.point))
            # assaing point to neuron
            self.neurons[tcn].append(point)

    def move_centers(self):
        """
        Deletes assing neurons posion to mean of therir points
        """
        for old_pos, points in self.neurons.items():
            # Don't do anything with dead neurons
            if(len(points) == 0):
                break
            # compute new posion
            x, y = zip(*points)
            avg_x = np.mean(x)
            avg_y = np.mean(y)
            # delete old neuron
            self.neurons.pop(old_pos)
            # add new
            self.neurons[(avg_x, avg_y)] = []

    def iter_once(self):
        self.match_points_to_neurons()
        self.move_centers()
