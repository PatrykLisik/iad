import csv
import sys

import numpy as np

from RBF import RBF


def MSE(f, intput, output):
    ans = 0
    for i in range(len(output)):
        ans += ((f(intput[i]) - output[i])**2)
    return np.sum(ans) / len(output)


def getData(intput):
    reader = csv.reader(intput)
    outX = []
    outY = []
    for row in reader:
        i = list(map(float, row[0].split(" ")))
        outX.append(i[0:-1])
        outY.append(i[-1])
    return outX, outY


# train set
train_input_list = []
train_target_list = []

intput_file = open(sys.argv[1], "r+")
train_input_list, train_target_list = getData(intput_file)
intput_file.close()

# test set
test_input_list = []
test_target_list = []

intput_file = open(sys.argv[2], "r+")
test_input_list, test_target_list = getData(intput_file)
intput_file.close()

number_of_iteration = 60
input_nodes = 1
output_nodes = 1
learningrate = 0.01
c_range = [min(train_input_list), max(train_input_list)]
sig_range = [0.6, 0.6]

head_line = ("Ilosc_neurnow", "MSE_AVG_train",
             "MSE_STD_train", "MSE_AVG_test", "MSE_STD_test")
results = []
results.append(head_line)

for h_nodes in range(1, 42, 5):
    hidden_nodes = h_nodes
    test_err = []
    train_err = []
    for _ in range(10):
        nn = RBF(input_nodes, hidden_nodes, output_nodes,
                 learningrate, c_range, sig_range)
        nn.set_up_centers_from_vec(train_input_list)
        for i in range(number_of_iteration):
            for input, outout in zip(train_input_list, train_target_list):
                nn.train_lin(input, outout)
        f = nn.query
        error_train = MSE(f, train_input_list, train_target_list)
        error_test = MSE(f, test_input_list, test_target_list)
        test_err.append(error_test)
        train_err.append(error_train)
    test_avg = np.mean(test_err)
    train_avg = np.mean(train_err)
    test_std = np.std(test_err, ddof=1)
    train_std = np.std(train_err, ddof=1)
    line = (h_nodes, train_avg, train_std, test_avg, test_std)
    print(line)
    results.append(line)

with open("results_3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
