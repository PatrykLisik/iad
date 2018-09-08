#!/usr/bin/env python3
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.ticker import OldScalarFormatter, LogFormatter, ScalarFormatter
from matplotlib import ticker
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from Perceptron import Perceptron
import csv
import sys


def MSE(f, intput, output):
    """
    Mean squared error
    """
    ans = 0
    for i in range(len(output)):
        ans += ((f(intput[i])-output[i])**2)
    return ans/len(output)


def getData(intput):
    reader = csv.reader(intput)
    outX = []
    outY = []
    for row in reader:
        i = list(map(float, row[0].split(";")))
        outX.append(i[0:-1])
        outY.append(i[-1])
    return outX, outY


def genName(intputFileName):
    noExt = intputFileName.split(".")[0]
    return "quality"+noExt[-1]+".png"


def myticks(x, pos):
    if x <= 0:
        return "$0$"
    exponent = int(np.log10(x))
    coeff = x/10**exponent
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


if __name__ == '__main__':
    intput_file = open(sys.argv[1], "r+")
    PointsX, PointsY = getData(intput_file)
    intput_file.close()

    perceptron = Perceptron(len(PointsX[0]))

    error = 1
    err_y = []
    err_x = []
    iter_number = 0
    min_error_treshold = 10**(-4)
    while error > min_error_treshold:
        error = MSE(perceptron, PointsX, PointsY)
        for x, y in zip(PointsX, PointsY):
            perceptron.updateWeigths(x, y)

        if iter_number%1000==0:
            err_y.append(error)
            err_x.append(iter_number)
            print(iter_number, " ", error)
        iter_number += 1
        # print to stdout


    # print to file
    print(perceptron.w0[0], file=open(sys.argv[2], "w"))
    for i in perceptron.W:
            # output file name is in second argument
        for j in i:
            print(j, file=open(sys.argv[2], "a"))

    # plot chart
    plotChart(err_x, err_y, genName(sys.argv[1]))
