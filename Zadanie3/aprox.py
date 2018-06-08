import copy
import csv
import sys

import matplotlib.pyplot as plt
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


def flat_list(d_list):
    return list(np.array(d_list).flat)


def plotChart(train_x, train_y, test_x, test_y, func_arry, title):
    # Set up plot
    plt.xlabel("x", fontsize="xx-large")
    plt.ylabel("y", fontsize="xx-large")
    train_x = flat_list(train_x)
    train_y = flat_list(train_y)
    test_x = flat_list(test_x)
    test_y = flat_list(test_y)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    # Plot train points
    plt.scatter(train_x, train_y, color='black',
                alpha=0.5, label="Punkty terningowe")
    # Plot test points
    plt.scatter(test_x, test_y, color='blue',
                alpha=0.1, label="Punkty testowe")
    # Plot given networks
    func_x = np.arange(min(train_x) - 1, max(train_x) + 1, 0.001)
    for iter, nn in func_arry.items():
        func_y = []
        for i in func_x:
            func_y.append(nn.query([i]))
        ax.plot(func_x, func_y, label="Epoki nauki {0}".format(iter))

    plt.grid()
    plt.title(title)
    # Legend
    plt.legend(loc='upper center',
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(title + ".png")


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

# networks to plot
networks = {}
number_of_iteration = 600
iter_to_plot = np.linspace(0, number_of_iteration, 6)


input_nodes = 1
hidden_nodes = 20
output_nodes = 1
learningrate = 0.1
bias = 1
momentum = 0

nn = RBF(input_nodes, hidden_nodes, output_nodes)


for i in range(number_of_iteration):
    f = nn.query
    error_train = MSE(f, train_input_list, train_target_list)
    error_test = MSE(f, test_input_list, test_target_list)
    if i % 3 == 0:
        print("Iteracja: ", i)
        print("error_train: ", error_train)
        print("error_test: ", error_test)
        print("hidden_nodes: ", hidden_nodes)
    for j in range(len(train_input_list)):
        nn.train(train_input_list[j], train_target_list[j])
    if i in iter_to_plot:
        n = copy.deepcopy(nn)
        networks[i] = n

plotChart(train_input_list, train_target_list,
          test_input_list, test_target_list,
          networks,
          "Stany pośrednie: plik:{1}, Epoki nauki:{0} Neurony: {2}"
          .format(number_of_iteration, str(sys.argv[1]), hidden_nodes))
