#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from NeutralNetwork import NeutralNetwork

# Mean squared error


def MSE(f, intput, output):
    ans = 0
    for i in range(len(output)):
        ans += ((f(intput[i]) - output[i])**2)
    return ans / len(output)


def getData(intput):
    reader = csv.reader(intput)
    outX = []
    outY = []
    for row in reader:
        i = list(map(float, row[0].split(" ")))
        outX.append(i[0:-1])
        outY.append(i[-1])
    return outX, outY


def genName(intputFileName):
    noExt = intputFileName.split(".")[0]
    return "quality" + noExt[-1] + ".png"


def myticks(x, pos):
    if x <= 0:
        return "$0$"
    exponent = int(np.log10(x))
    coeff = x / 10**exponent
    return r"${:2.1f} \times 10^{{ {:2d} }}$".format(coeff, exponent)


def plotChart(x, y, out):
        # Set up plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    plt.grid()
    plt.title(genName(sys.argv[1]))
    # plt.ylim([0,0.02])
    # plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of iterations", fontsize="xx-large")
    plt.ylabel("Mean squared error", fontsize="xx-large")
    ax.plot(x, y, color='purple')
    plt.savefig(out)


input_list = []
target_list = []

intput_file = open(sys.argv[1], "r+")
input_list, target_list = getData(intput_file)
intput_file.close()


input_nodes = len(input_list[0])
hidden_nodes = 3
output_nodes = 1
learningrate = 0.1
bias = True
nn = NeutralNetwork(input_nodes, hidden_nodes,
                    output_nodes, learningrate, bias)


i = 0
error_test = 10
ErrorX = []
ErrorY = []
while error_test > 9.898:
    for j in range(len(input_list)):
        nn.train(input_list[j], target_list[j])
        f = nn.query
        error_test = MSE(f, input_list, target_list)[0][0]
        ErrorX.append(i)
        ErrorY.append(error_test)
    if i % 50 == 0:
        print(error_test)
    i += 1

for i in range(len(input_list)):
    print("Expected: ", input_list[i], "\nNetwork: ", nn.query(input_list[i]))
plotChart(ErrorX, ErrorY, "wykres {}-{}-{}-bias-{}.png"
          .format(intput_nodes, hidden_nodes, output_nodes, bias))
