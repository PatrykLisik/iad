#!/usr/bin/env python3
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from NeutralNetwork import NeutralNetwork
from Functions import MSE, getData
import numpy as np
from matplotlib.pyplot import cm
from matplotlib import ticker
import matplotlib.pyplot as plt


def myticks(x,pos):
    if x <= 0:
         return "$0$"
    exponent = int(np.log10(x))
    coeff = x/10**exponent
    return r"${:2.1f} \times 10^{{ {:2d} }}$".format(coeff,exponent)
def plotChart(x,y,out):
        #gen colors
        colors=iter(cm.rainbow(np.linspace(0,1,len(x))))
        #Set up plot
        fig=plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
        plt.grid()
        plt.title(genName(sys.argv[1]))
        #plt.ylim([0,0.02])
        #plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of iterations",fontsize="xx-large")
        plt.ylabel("Mean squared error", fontsize="xx-large")
        for i in range(len(x)):
            ax.plot(x[i],y[i])
        plt.savefig(out)

input_list = []
target_list = []
input_file = open(sys.argv[1], "r+")
input_list=getData(input_file)
target_list=input_list
input_file.close()


ErrorXList=[]
ErrorYList=[]

for k in range(1,4):
    input_nodes = len(input_list[0])
    hidden_nodes = k
    output_nodes = len(target_list[0])
    learningrate = 0.1
    nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes, learningrate,0)


    i=0
    error_test=10;
    ErrorX=[]
    ErrorY=[]
    while i<10**5:
        for j in range(len(input_list)):
            nn.train(input_list[j],target_list[j])
            f=nn.query
            error_test=MSE(f,input_list,target_list)
            ErrorX.append(i);
            ErrorY.append(error_test)
        if i%10000==0:
            print("Number of hidden nodes: {0} \n Error: {1}".format(k,error_test))
            print(f(input_list[0]))
        i+=1
    ErrorXList.append(ErrorX)
    ErrorYList.append(ErrorY)

for i in range(len(input_list)):
    print("Expected: ",input_list[i],"\nNetwork: ",nn.query(input_list[i]))
plotChart(ErrorXList,ErrorYList,"wykres 4-3-4-nb.png")
