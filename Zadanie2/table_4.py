from SOM.K_means import K_menas as KM
from SOM.functions import quantization_error2
from points_distributions import circumference_dist, triangle_dist
import csv
import numpy as np
import copy


def deadNumber(data):
    data_trans = np.swapaxes(data, 1, 0)
    ans = 0
    # print("data", data)
    for neuron_history in data_trans:
        # test if all are the same
        ans += int((neuron_history[1:] == neuron_history[:-1]).all())
    return ans


results = []
results.append(["LR", "DEAD_MEAN", "DEAD_STD", "EQ_MIN", "MSQE", "STD"])

set = []
set.extend(circumference_dist([0, 0], 7, 400))
set.extend(triangle_dist([5, 5], [5, 8], [7, 8], 400))

neuron_number = 20
lrs = list(np.linspace(0.001, 0.4, 10))
for lr in lrs:
    one_iter = [lr]
    QErr = []
    deadNeuron = []
    for _ in range(100):
        kn = KM(neuron_number, set, lr=lr)
        neurons_in_time = []
        # Teach networks
        for _ in range(9):
            kn.iter_once()
            neuron_cpy = copy.deepcopy(list(kn.getNeurons()))
            neurons_in_time.append(neuron_cpy)
        QErr.append(quantization_error2(kn.getNeurons(), set))
        deadNeuron.append(deadNumber(neurons_in_time))
    min_err = min(QErr)
    kn_errMean = np.mean(QErr)
    kn_STD = np.std(QErr, ddof=1)
    dead_mean = np.mean(deadNeuron)
    dead_STD = np.std(deadNeuron, ddof=1)
    one_iter.extend([dead_mean, dead_STD, min_err, kn_errMean, kn_STD])

    results.append(one_iter)
    print("one_iter: ", one_iter)
with open("double_results_kmeans.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
