from SOM.K_means import K_means as KM
from SOM.functions import quantization_error3 as q_Err
from points_distributions import circumference_dist, triangle_dist
import csv
import numpy as np
import copy
from SOM.functions import deadNumber
from points_distributions import points


results = []
results.append(["sq_param", "DEAD_MEAN", "DEAD_STD", "EQ_MIN", "MSQE", "STD"])

set = []
set.extend(circumference_dist([-2, -2], 5, 400))
set.extend(triangle_dist([5, 5], [5, 8], [7, 8], 400))

neuron_number = 20
sq_params = list(range(4, 25, 3))
for points_name, set in points.items():
    for sq_param in sq_params:
        one_iter = [sq_param]
        QErr = []
        deadNeuron = []
        for _ in range(100):
            kn = KM(neuron_number, set, low=-sq_param, high=sq_param)
            neurons_in_time = []
            # Teach networks
            for _ in range(9):
                kn.iter_once()
                neuron_cpy = copy.deepcopy(kn.getNeurons())
                neurons_in_time.append(neuron_cpy)
            QErr.append(q_Err(kn.getNeurons(), set))
            deadNeuron.append(deadNumber(neurons_in_time))
        min_err = min(QErr)
        kn_errMean = np.mean(QErr)
        kn_STD = np.std(QErr, ddof=1)
        dead_mean = np.mean(deadNeuron)
        dead_STD = np.std(deadNeuron, ddof=1)
        one_iter.extend([dead_mean, dead_STD, min_err, kn_errMean, kn_STD])

        results.append(one_iter)
        print(points_name + "one_iter: ", one_iter)
    with open(points_name + "_results_kmeans.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
