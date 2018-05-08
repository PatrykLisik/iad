from SOM.K_means import K_menas as KM
from SOM.functions import quantization_error2
from points_distributions import circumference_dist, triangle_dist
import csv
import numpy as np

results = []
results.append(["LR", "MIN", "MSQE", "STD"])

set = []
set.extend(circumference_dist([3, 3], 3, 400))
#set.extend(triangle_dist([0, 0], [0, 4], [-4, 0], 400))

neuron_number = 20
lrs = list(np.linspace(0.001, 0.4, 5))
for lr in lrs:
    one_iter = [lr]
    QErr = []

    for _ in range(50):
        kn = KM(neuron_number, set, lr=lr)
        # Teach networks
        for _ in range(8):
            kn.iter_once()
        QErr.append(quantization_error2(kn.getNeurons(), set))
    min_err = min(QErr)
    kn_errMean = np.mean(QErr)
    kn_STD = np.std(QErr, ddof=1)
    one_iter.extend([min_err, kn_errMean, kn_STD])

    results.append(one_iter)
    print("one_iter: ", one_iter)
with open("single_results_kmeans.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
