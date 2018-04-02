#!/usr/bin/env python3
'''Arguments
-1 train_set
-2 test_set'''

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from NeutralNetwork import NeutralNetwork
from Functions import MSE,getData,myticks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pyplot import cm

def plotChart(data,out):
    colors=iter(cm.Set1(np.linspace(0,1,len(data))))
    #colors=iter(cm.Dark2(np.linspace(0,1,len(data))))
    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    plt.grid()
    plt.yscale("log")
    plt.xlabel("Ilość iteracji",fontsize="xx-large")
    plt.ylabel("Błąd śreniokwadratowy", fontsize="xx-large")
    for n_of_neurons,lists in data.items():
        c=next(colors)
        ax.plot(lists[0],lists[1],color=c,
                label="Ilość neuronow {0}-błąd dla danych treningowych".format(n_of_neurons))
        ax.plot(lists[0],lists[2],color=c,linestyle=":",
                label="Ilość neuronow {0}-błąd dla danych testowych".format(n_of_neurons))
    legend = plt.legend(loc='upper center',
              ncol=3, fancybox=True, shadow=True)
    legend.get_frame()
    plt.title(out)
    plt.savefig(out+".png")




#train set
train_input_list = []
train_target_list = []

intput_file = open(sys.argv[1], "r+")
train_input_list,train_target_list=getData(intput_file)
intput_file.close()

#test set
test_input_list = []
test_target_list = []

intput_file = open(sys.argv[2], "r+")
test_input_list,test_target_list=getData(intput_file)
intput_file.close()

number_of_iteration=3*10**3
errors_to_plot={}
learningrate = 0.1
bias=1;
momentum=0
for k in [1,5,19]:
    i=0
    error_train=10;
    error_test=10
    ErrorX=[]
    ErrorY_train=[]
    ErrorY_test=[]
    input_nodes = 1
    hidden_nodes = k
    output_nodes = 1
    activation_function_output=lambda x: x
    dactivation_function_output=lambda x: 1
    nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes,
    learningrate,bias,momentum,activation_function_output,
    dactivation_function_output)
    while i<number_of_iteration:
        f=nn.query
        error_train2=error_train
        error_train=MSE(f,train_input_list,train_target_list)
        error_test=MSE(f,test_input_list,test_target_list)
        if i%100==0:
            '''print("Iteracja: ",i)
            print("error_train: ",error_train)
            print("error_test: ",error_test)
            print("hidden_nodes: ",hidden_nodes)'''
        for j in range(len(train_input_list)):
            nn.train(train_input_list[j],train_target_list[j])
        ErrorX.append(i);
        ErrorY_train.append(error_train)
        ErrorY_test.append(error_test)
        i+=1
    errors_to_plot[k]=[ErrorX,ErrorY_train,ErrorY_test]
plotChart(errors_to_plot,
          "Błędy: plik:{1}, Iteracje:{0} Learningrate: {2}".
          format(number_of_iteration,str(sys.argv[1])),learningrate)
