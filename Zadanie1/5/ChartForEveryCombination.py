#!/usr/bin/env python3
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from NeutralNetwork import NeutralNetwork
from Functions import MSE
import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt

def plotChart(dict_data,title):
    lr=list(dict_data.keys())
    val=list(dict_data.values())
    ind = np.arange(len(val))
    width = 0.35
    #Set up plot

    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind,val,width,label="% Rozpoznanych przypadków")
    ax.set_xticks(ind )
    ax.set_xticklabels(lr)
    plt.xlabel("Ilosć neuronów")
    #plt.ylabel("% Rozpoznanych przypadków")
    plt.grid()
    plt.title(title)
    #Legend
    plt.legend(loc='upper center',
              ncol=3, fancybox=True, shadow=True)
    plt.savefig(title+".png")

def getDataSep(intput):
    reader = csv.reader(intput)
    out1=[]
    out2=[]
    out3=[]
    out4=[]
    ans=[]
    ans_trans={1:[1,0,0],
               2:[0,1,0],
               3:[0,0,1]}

    for row in reader:
        i = list(map(float,row[0].split(" ")))
        out1.append(i[0])
        out2.append(i[1])
        out3.append(i[2])
        out4.append(i[3])
        ans.append(ans_trans[i[4]])

    return out1,out2,out3,out4,ans

def flatter(input):
    if input>0.5:
        return 1
    else:
        return 0

def clas_test(ans, target):
    return (ans==target).all()

#train set
    out1=[]
    out2=[]
    out3=[]
    out4=[]
    ans=[]
intput_file = open(sys.argv[1], "r+")
out1,out2,out3,out4,ans=getDataSep(intput_file)
intput_file.close()
out=[out1,out2,out3,out4]

test={1:30,5:60,9:80,12:96}
plotChart(test,"title")
iterable=[0,1,2,3]
#input_numbres
for input_neuron_number in range(1,5):

    list_list=list(itertools.combinations(iterable, input_neuron_number))
    print("input_neuron_number",input_neuron_number)
    print("len ",len(list_list))
    for ll in list_list:
        #prepare data
        train_set=[]
        for k in range(len(out1)):
            #set of points to train
            train_tab=[]
            for kk in ll:
                ##list kk, k element
                train_tab.append(out[kk][k])
            train_set.append(train_tab)

        data_for_single_chart={}
        for hnodes in range(1,18,4):
            iter=0
            input_nodes = input_neuron_number
            hidden_nodes = hnodes
            output_nodes = 3
            learningrate = 0.1
            nn = NeutralNetwork(input_neuron_number, hidden_nodes, output_nodes)
            while iter<3*10**3:
                #train single epoch
                #k-point to train
                for k in range(len(train_set)):
                    nn.train(train_set[k],ans[k])
                f=nn.query
                error=MSE(f,train_set,ans)
                '''if iter%1000==0:
                    print("ERROR: ",error)
                    print("LL: ",ll)
                    print("iter: ",iter)
                    print("hnodes: ",hnodes)'''
                iter+=1
            good=0
            for test in range(len(train_set)):
                t=np.round(nn.query(train_set[test]).T,0)
                good+=clas_test(ans[test],t)
                '''print("Network: ",t)
                print("Ans:",ans[test])
                print("Good?",clas_test(ans[test],t),"\n\n")'''
            percent=good/len(train_set)*100
            print("hnodes: ",hnodes)
            print("\nPERCENT: ",percent,"\n")
            data_for_single_chart[hnodes]=percent
        plotChart(data_for_single_chart,"Ilosć wejść-{0},kolumny-{1}".format(input_neuron_number,str(ll)))
