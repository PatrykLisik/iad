import itertools
import math

import numpy as np


class KohonenNetwork:
    """Implementation of n-dimmensional kohonen network"""

    def __init__(self, dim_newtork, dist_func_points, net_dist_to_lr,
                 points_to_aprox):
        """
        Args:
            dinm_network: number of dimmensions of network organization
            dist_func: callable objest that takes two points of from
                        points_to_aprox and returns distance beetwen them
            net_dist_to_lr: callable object that takes posion of two neurons
                            and returns leraning_rate_multiplayer. Throu this
                            argument WTA nad WTM approach can be achived
            points_to_aprox: list of points to perform approximation on
        """
        pass

    def iter_once(self):
        """
        Performs one iteration of approximation of network
        Returns:
                Dictioanry of tuple to tuple
                {(Pos_in_netowrk):(Pos_in_space)}
        """
        pass

    def _random_point(self, n):
        """
        Generete random point in n-dimensional space
        Returns:
                n- element tuple of random 32-bit floats
        """
        return tuple(np.random.rand(n, 1))

    def _getLR(self, posNWinner, posNOther):
        """
        Private method that compute how much posNOther neuron should be moved
        toward point
        Learnning rate sholud decreasing in iterations
        Output sholud be the higest to case where posNWinner==posNOther is true
        and be the smallest where distance beteetwen neurons in
        network is the higest
        Args:
            posNWinner: tuple of n floats number desricbing posion in network
                        of winner-neuron(neuron which is the closest to point)
            posNOther: tuple of n floats number desricbing posion in network of
                        neuron which ins't the closest
        Returns:
            Float number grater than 0
        """
        pass

    def _genreteStartNeurons(self, enctrance_number, dim_number):
        """
        Private method that generate random starting points of network
        Args:
            enctrance_number: number of entrances in output
            dim_number: number dimmensions of neuron network
        Return:
                Dictioanry of tuple to tuple
                {(Pos_in_netowrk):(Pos_in_space)}
        """

        ans = {}
        entry_counter = 0
        mx = math.ceil(math.pow(enctrance_number, 1 / dim_number))
        # x is tuple of params in n-dimensional space
        for x in itertools.product(range(mx), repeat=dim_number):
            if entry_counter == enctrance_number:
                break
            ans[x] = self._random_point(dim_number)
            entry_counter += 1
        return ans
