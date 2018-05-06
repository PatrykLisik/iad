from .Self_organizing_map import Self_organizing_map
import operator
from collections import OrderedDict
from .functions import Euklides_dist as E_dist
from .functions import GNF
import numpy as np


class Neuron_gas(Self_organizing_map):
    """Implementation neuron gas"""

    def __init__(self, points_number, points_to_aprox, neighborhood_radius=2,
                 net_dist_to_lr=GNF, dist_func_points=E_dist):
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
        self.lazy_numer = int(np.sqrt(points_number))
        super().__init__(points_number, neighborhood_radius, 1,
                         dist_func_points, net_dist_to_lr,
                         points_to_aprox, self.lazy_numer)
        self.lr = 0.2

    # override
    def _update_neurons_space_posisions(self, winner, point):
        """
        Update posioson of all neurons
        Function takes winner and corresponding point

        Args:
            winner - tuple describing nuron in network
            point - posion of point in space
        """
        # generate sorted dict
        # sort function is fucntion which gives distance
        # between point and neruon
        def dist_from_point(p): return self.dist_points(point, p[1])
        sorted_neurons = OrderedDict(sorted(
            self.neurons.items(),
            key=dist_from_point))

        for number, pos_net in enumerate(sorted_neurons.keys()):
            lr = self._getLR((0,), (number,))
            # distance beteetwen point and neuron in every dimmension
            update_vals = tuple(map(operator.sub, point,
                                    self.neurons[pos_net]))
            # every value is multiplied elementwise by lr
            update_vals = tuple(map(lr.__mul__, update_vals))
            # elementwise add on tuple
            self.neurons[pos_net] = tuple(
                map(operator.add, update_vals, self.neurons[pos_net]))
