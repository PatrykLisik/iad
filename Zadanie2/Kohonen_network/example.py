from Kohonen_network import KohonenNetwork
from functons import RNF, Euklides_dist, getPointsInCircle, print_net


points_number = 5
dimm = 2
aprox_set = getPointsInCircle([10, 10], 1, 100)
kn = KohonenNetwork(points_number, dimm, Euklides_dist, RNF, aprox_set)

n = kn.neurons
print_net(n)
for i in range(10):
    kn.iter_once()
    n = kn.neurons
    print(kn.tired._data)
    print_net(n)
