#!/usr/bin/env python
import inspect
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.pyplot import cm

from common import getDataSep, recognitionPerc
from Functions import MSE, myticks
from ready import RBF


def plotChart(data, title):
    # colors=iter(cm.Set1(np.linspace(0,1,len(data))))
    colors = iter(cm.Dark2(np.linspace(0, 1, len(data))))
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    plt.grid()
    # plt.yscale("log")
    plt.xlabel("Ilość iteracji", fontsize="xx-large")
    plt.ylabel("%Rozpoznanych przypadków", fontsize="xx-large")
    plt.ylim([0, 100])

    # lists=[iter_n,train_percentage,test_percentage]
    for n_of_neurons, lists in data.items():
        c = next(colors)
        ax.plot(lists[0], lists[1], color=c,
                label="Ilość neuronow {0}".format(n_of_neurons))
        ax.plot(lists[0], lists[2], color=c, linestyle=":",
                label="Ilość neuronow {0}".format(n_of_neurons))
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                        ncol=3, fancybox=True, shadow=True)
    legend.get_frame()
    plt.title(title)
    plt.savefig(title + ".png", bbox_extra_artists=(legend,),
                bbox_inches='tight')


# train set
out_train = []
ans_train = []
intput_file = open(sys.argv[1], "r+")
out_train, ans_train = getDataSep(intput_file)
intput_file.close()

# test set
out_test = []
ans_test = []
intput_file = open(sys.argv[2], "r+")
out_test, ans_test = getDataSep(intput_file)
intput_file.close()

iterable = [0, 1, 2, 3]
# input_numbres
for input_neuron_number in range(1, 5):

    list_list = list(itertools.combinations(iterable, input_neuron_number))
    print("input_neuron_number", input_neuron_number)
    print("len ", len(list_list))
    for ll in list_list:
        # prepare data
        train_set = []
        test_set = []
        for k in range(len(out_train[0])):
            # set of points to train
            train_tab = []
            for kk in ll:
                # list kk, k element
                train_tab.append(out_train[kk][k])
            train_set.append(train_tab)
        for k in range(len(out_test[0])):
            # set of points to train
            test_tab = []
            for kk in ll:
                # list kk, k element
                test_tab.append(out_test[kk][k])
            test_set.append(test_tab)

        data_for_single_chart = {}
        for hnodes in range(1, 18, 4):
            iter_n = 0
            input_nodes = input_neuron_number
            hidden_nodes = hnodes
            output_nodes = 3
            learningrate = 0.001
            nn = RBF(input_neuron_number,
                     hidden_nodes, output_nodes)
            nn.set_up_centers(train_set)
            percent_train_tab = []
            percent_test_tab = []
            iter_tab = []
            while iter_n < 60:
                # train single epoch
                # k-point to train
                for k in range(len(train_set)):
                    nn.train(train_set[k], ans_train[k])
                f = nn.query
                error = MSE(f, train_set, ans_train)
                iter_n += 1

                # Compute error
                percent_train = recognitionPerc(train_set, ans_train, nn)
                percent_test = recognitionPerc(test_set, ans_test, nn)
                # Paste to table
                percent_train_tab.append(percent_train)
                percent_test_tab.append(percent_test)
                iter_tab.append(iter_n)
            print("hnodes: ", hnodes)
            data_for_single_chart[hnodes] = [
                iter_tab, percent_train_tab, percent_test_tab]
        print("Chart plotting")
        plotChart(data_for_single_chart,
                  "Ilosć wejść-{0},kolumny-{1}"
                  .format(input_neuron_number, str(ll)))
