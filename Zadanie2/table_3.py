from SOM.Neuron_gas import Neuron_gas
from SOM.Kohonen_network import Kohonen_network
from SOM.functions import RNF, GNF, Euklides_dist
from SOM.functions import quantization_error3 as q_err
from points_distributions import circumference_dist, triangle_dist, cirlce_dist
from points_distributions import square_dist
import csv
import numpy as np

results_koh = []
results_gas = []
results_gas.append(["LR", "Tired number", "MSQE", "STD"])
results_koh.append(["LR", "Tired number", "MSQE", "STD"])

set = []
set.extend(circumference_dist([3, 3], 3, 400))
set.extend(triangle_dist([0, 0], [0, 4], [-4, 0], 400))

neuron_number = 20
tireds = [0] * 5 + list(range(0, 10, 2))
lrs = list(np.linspace(0.001, 0.4, 5)) + [0.2] * 5
for lr, tired in zip(lrs, tireds):
    one_iter_koh = [lr, tired]
    one_iter_gas = [lr, tired]
    QErr_koh = []
    QErr_gas = []

    for _ in range(100):
        kn = Kohonen_network(neuron_number, set, lr=lr, lazy_numer=tired)
        gn = Neuron_gas(neuron_number, set, lr=lr, lazy_numer=tired)
        # Teach networks
        for _ in range(20):
            kn.iter_once()
            gn.iter_once()
        QErr_koh.append(q_err(kn.neurons.values(), set))
        QErr_gas.append(q_err(gn.neurons.values(), set))
    kn_errMean = np.mean(QErr_koh)
    gn_errMean = np.mean(QErr_gas)
    kn_STD = np.std(QErr_koh, ddof=1)
    gn_STD = np.std(QErr_gas, ddof=1)
    one_iter_koh.extend([kn_errMean, kn_STD])
    one_iter_gas.extend([gn_errMean, gn_STD])

    results_koh.append(one_iter_koh)
    print("Koh: ", one_iter_koh)
    results_gas.append(one_iter_gas)
    print("Gas: ", one_iter_gas)
with open("results_koh.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results_koh)
with open("results_gas.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results_gas)
