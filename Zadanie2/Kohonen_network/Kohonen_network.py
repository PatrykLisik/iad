import itertools
import math
import sys
import operator
from Fixed_size_queue import Fixed_size_queue
import numpy as np


class KohonenNetwork:
    """Implementation of n-dimmensional kohonen network"""

    def __init__(self, points_number, dim_network, dist_func_points,
                 net_dist_to_lr, points_to_aprox):
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
        # assume that all points have same number of dimmensions
        dimm_space = len(points_to_aprox[0])
        self.neurons = self._genreteStartNeurons(
            points_number, dim_network, dimm_space)
        # compute distnce between points in space
        self.dist_points = dist_func_points
        # Convert distnce to lr multipler
        self.dist_net_to_lr = net_dist_to_lr
        # compute distance beetwen points in network
        self.dist_net = dist_func_points
        # Learnning rate
        self.lr = 0.5
        # Iteration counter
        self.iter_count = 0
        # Points list to perfrom operation on
        self.points_to_aprox = points_to_aprox
        # neighborhood radius is temporarily hardcoded to 0
        self.R = 1
        # tired neurons
        self.tired = Fixed_size_queue(2)

    def iter_once(self):
        """
        Iterate over every given point and adjust neurons
        """
        for point in self.points_to_aprox:
            colsest = self._find_closest_neuron(point)
            self._update_neurons_space_posisions(colsest, point)
        self._updateLr()

    def _find_closest_neuron(self, point):
        """
        Args:
            point: n-element tuple that represents point in space
        Returns:
            k-elemnt tuple that reprezents neuron position in neuron network
        """
        # tuple to returns
        ans = []
        # init min with its max value
        min_dist = sys.float_info.max
        for pos_net, pos_space in self.neurons.items():
            # skip if neuron won last time
            if(self.tired.contains(pos_net)):
                continue
            # distance beteween particular neuron and point
            dist = self.dist_points(pos_space, point)
            if min_dist > dist:
                min_dist = dist
                ans = pos_net
        # add new winner
        self.tired.append(ans)
        return ans

    def _update_neurons_space_posisions(self, winner, point):
        """
        Update posioson of all neurons
        Function takes winner and corresponding point

        Args:
            winner - tuple describing nuron in network
            point - posion of point in space
        """
        for pos_net, pos_space in self.neurons.items():
            lr = self._getLR(winner, pos_net)
            # distance beteetwen point and neuron in every dimmension
            update_vals = tuple(map(operator.sub, point,
                                    self.neurons[pos_net]))
            # every value is multiplied elementwise by lr
            update_vals = tuple(map(lr.__mul__, update_vals))
            # elementwise add on tuple
            self.neurons[pos_net] = tuple(
                map(operator.add, update_vals, self.neurons[pos_net]))

    def _random_point(self, n):
        """
        Generete random point in n-dimensional space
        Returns:
                n - element tuple of random 32-bit floats
        """
        # bounds are temporarily hardcoded to -10,10
        return tuple(np.random.uniform(low=-10, high=10, size=(n)))

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
        dist = self.dist_net(posNWinner, posNOther)
        ans = self.lr * self.dist_net_to_lr(self.R, dist)
        return ans

    def _updateLr(self):
        """
        Private method that decrese learning rate
        """

        # This migth be suboptimal function
        self.lr /= 1.1

    def _genreteStartNeurons(self, enctrance_number, dim_number_net,
                             dim_number_space):
        """
        Private method that generate random starting points of network
        Args:
            enctrance_number: number of entrances in output
            dim_number_net: number dimmensions of neuron network
        Return:
                Dictioanry of tuple to tuple
                {(Pos_in_netowrk):(Pos_in_space)}
        """

        ans = {}
        entry_counter = 0
        mx = math.ceil(math.pow(enctrance_number, 1 / dim_number_net))
        # x is tuple of params in n-dimensional space
        for x in itertools.product(range(mx), repeat=dim_number_net):
            if entry_counter == enctrance_number:
                break
            ans[x] = self._random_point(dim_number_space)
            entry_counter += 1
        return ans
