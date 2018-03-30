#!/usr/bin/env python3
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from NeutralNetwork import NeutralNetwork
from Functions import MSE,myticks,getAllData
import numpy as np
from matplotlib.pyplot import cm
from matplotlib import ticker
import matplotlib.pyplot as plt


def plotChart(x,y,out):
        fig=plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
        plt.grid()
        plt.title(out)
        plt.yscale("log")
        plt.xlabel("Ilość iteracji",fontsize="xx-large")
        plt.ylabel("Błąd śreniokwadratowy", fontsize="xx-large")
        for i in range(len(x)):
            ax.plot(x[i],y[i],label="Ilość neuronow {0}".format(i+1))
        legend = plt.legend(loc='upper center',
                  ncol=3, fancybox=True, shadow=True)
        legend.get_frame()
        plt.savefig(out)


#BEGIN OF THE SCRIPT
input_list = []
target_list = []
input_file = open(sys.argv[1], "r+")
input_list=getAllData(input_file)
target_list=input_list
input_file.close()

#bias
for bias_loop,sw in {1:"on",0:"off"}.items():
    ErrorXList=[]
    ErrorYList=[]
    #Number of neurons
    for k in range(1,4):
        input_nodes = 4
        hidden_nodes = k
        output_nodes = 4
        learningrate = 0.1
        bias=bias_loop
        nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes, learningrate,bias)
        i=0
        ErrorX=[]
        ErrorY=[]
        while i<10**5:
            for j in range(len(input_list)):
                nn.train(input_list[j],target_list[j])
                f=nn.query
                error_test=MSE(f,input_list,target_list)
                ErrorX.append(i);
                ErrorY.append(error_test)
            #Debug info
            '''if i%10000==0:
                print("i:",i,"\nNumber of hidden nodes: {0} \n Error: {1}".format(k,error_test))
                for ki in range(len(input_list)):
                    print("Expected: ",input_list[ki],"\nNetwork: ",nn.query(input_list[ki]).T)'''
            i+=1
        ErrorXList.append(ErrorX)
        ErrorYList.append(ErrorY)
        '''for i in range(len(input_list)):
            print("Expected: ",input_list[i],"\nNetwork: ",nn.query(input_list[i]).T)'''

    plotChart(ErrorXList,ErrorYList,"Neurony-Bias-"+sw)
