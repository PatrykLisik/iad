from .Self_organizing_map import Self_organizing_map
import operator
from .functions import GNF
from .functions import Euklides_dist as E_dist


class Kohonen_network(Self_organizing_map):
    """Implementation neuron gas"""

    def __init__(self, points_number, points_to_aprox, neighborhood_radius=1,
                 net_dist_to_lr=GNF, dist_func_points=E_dist,
                 lazy_numer=None, lr=0.2, dim_network=1):
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
        if(lazy_numer is None):
            self.lazy_numer = int(points_number / 10) + 1
        else:
            self.lazy_numer = lazy_numer

        neighborhood_radius = 1
        super().__init__(points_number, neighborhood_radius, dim_network,
                         dist_func_points, net_dist_to_lr,
                         points_to_aprox, self.lazy_numer)

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
