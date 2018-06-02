from SOM.Neuron_gas import Neuron_gas
from SOM.Kohonen_network import Kohonen_network
from SOM.functions import quantization_error3 as q_err
from points_distributions import points
import csv
import numpy as np
from SOM.functions import deadNumber

results_koh = []
results_gas = []
header = ["LR", "Tired number", "MSQE", "STD", "DEAD_AVG", "DEAD_STD"]
results_gas.append(header)
results_koh.append(header)

data_sets = points

neuron_number = 20
tireds = [0] * 5 + list(range(0, 10, 2))
lrs = list(np.linspace(0.001, 0.4, 5)) + [0.2] * 5
for points_name, set in data_sets.items():
    for lr, tired in zip(lrs, tireds):
        one_iter_koh = [lr, tired]
        one_iter_gas = [lr, tired]
        QErr_koh = []
        QErr_gas = []
        dead_gas = []
        dead_koh = []

        for _ in range(10):
            kn = Kohonen_network(neuron_number, set, lr=lr, lazy_numer=tired)
            gn = Neuron_gas(neuron_number, set, lr=lr, lazy_numer=tired)
            neur_kn = []
            neur_gn = []
            # Teach networks
            for _ in range(5):
                kn.iter_once()
                neur_kn.append(kn.getNeurons())
                gn.iter_once()
                neur_gn.append(gn.getNeurons())
            QErr_koh.append(q_err(kn.neurons.values(), set))
            QErr_gas.append(q_err(gn.neurons.values(), set))
            dead_koh.append(deadNumber(neur_kn))
            dead_gas.append(deadNumber(neur_gn))
        kn_errMean = np.mean(QErr_koh)
        gn_errMean = np.mean(QErr_gas)
        kn_STD = np.std(QErr_koh, ddof=1)
        gn_STD = np.std(QErr_gas, ddof=1)
        one_iter_koh.extend([kn_errMean, kn_STD])
        one_iter_gas.extend([gn_errMean, gn_STD])
        one_iter_koh.append(np.mean(dead_koh))
        one_iter_koh.append(np.std(dead_koh, ddof=1))
        one_iter_gas.append(np.mean(dead_gas))
        one_iter_gas.append(np.std(dead_gas, ddof=1))

        results_koh.append(one_iter_koh)
        print("Koh: ", one_iter_koh)
        results_gas.append(one_iter_gas)
        print("Gas: ", one_iter_gas)
    with open(points_name + "_" + "results_koh.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_koh)
    with open(points_name + "_" + "results_gas.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_gas)
