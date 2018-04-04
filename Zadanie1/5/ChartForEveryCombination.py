#!/usr/bin/env python3
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from NeutralNetwork import NeutralNetwork
from Functions import MSE
import numpy as np
import itertools
import matplotlib.pyplot as plt


def plotChart(dict_data,title):
    neuron=list(dict_data.keys())
    val=list(dict_data.values())
    train=[]
    test=[]
    for i,j in val:
        train.append(i)
        test.append(j)
    ind = np.arange(len(val))
    width = 0.35
    #Set up plot

    fig=plt.figure()
    ax = fig.add_subplot(111)
    train_bar=ax.bar(ind,train,width,label="Zbiór treningowy")
    test_bar=ax.bar(ind+width,test,width,label="Zbiór testowy")
    ax.set_xticks(ind+width/2 )
    ax.set_xticklabels(neuron)
    plt.xlabel("Ilosć neuronów")
    plt.ylabel("% Rozpoznanych przypadków")
    plt.grid()
    plt.title(title)
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
            '%d' % int(height),
            ha='center', va='bottom')
    autolabel(train_bar)
    autolabel(test_bar)
    #Legend
    lgd=plt.legend()
    plt.savefig(title+".png",bbox_extra_artists=(lgd,), bbox_inches='tight')

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

    return [out1,out2,out3,out4],ans

def recognitionPerc(input,ans,nn):
    good_ans=0
    length=len(train_set)
    for test in range(length):
        t=np.round(nn.query(input[test]).T,0)
        good_ans+=clas_test(ans_train[test],t)
    return good_ans/length*100

def clas_test(ans, target):
    return (ans==target).all()

#train set
out_train=[]
ans_train=[]
intput_file = open(sys.argv[1], "r+")
out_train,ans_train=getDataSep(intput_file)
intput_file.close()

#test set
out_test=[]
ans_test=[]
intput_file = open(sys.argv[2], "r+")
out_test,ans_test=getDataSep(intput_file)
intput_file.close()

test={1:[20,10],
      5:[50,40],
      17:[96,70]}
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
        test_set=[]
        for k in range(len(out_train[0])):
            #set of points to train
            train_tab=[]
            test_tab=[]
            for kk in ll:
                ##list kk, k element
                train_tab.append(out_train[kk][k])
                test_tab.append(out_test[kk][k])
            train_set.append(train_tab)
            test_set.append(test_tab)

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
                    nn.train(train_set[k],ans_train[k])
                f=nn.query
                error=MSE(f,train_set,ans_train)
                iter+=1
            percent_train=recognitionPerc(train_set,ans_train,nn)
            percent_test=recognitionPerc(test_set,ans_test,nn)
            print("hnodes: ",hnodes)
            print("percent_train: ",percent_train)
            print("percent_test: ",percent_test,"\n\n")
            data_for_single_chart[hnodes]=[percent_train,percent_test]
        plotChart(data_for_single_chart,"Ilosć wejść-{0},kolumny-{1}".format(input_neuron_number,str(ll)))
