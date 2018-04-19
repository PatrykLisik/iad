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
import csv

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


def plotChart(nn,train,test,train_ans,test_ans,out):
    y_min=0
    y_max=7
    x_min=0.5
    x_max=9
    step=0.0015
    all_points=[]
    for x in np.arange(x_min,x_max,step):
        for y in np.arange(y_min,y_max,step):
            all_points.append([x,y])
            #print(all_points)
    green_x=[] # obj 1
    green_y=[] # obj 1
    blue_x=[] # obj 2
    blue_y=[] # obj 2
    red_x=[] #obj 3
    red_y=[] #obj 3

    ans_green_x=[]
    ans_green_y=[]
    ans_red_x=[]
    ans_red_y=[]
    ans_blue_x=[]
    ans_blue_y=[]
    for i in range(len(train_ans)):
        if train_ans[i]==[1, 0, 0]:
            ans_green_x.append(train[i][0])
            ans_green_y.append(train[i][1])
        if train_ans[i]==[0, 1, 0]:
            ans_blue_x.append(train[i][0])
            ans_blue_y.append(train[i][1])
        if train_ans[i]==[0, 0, 1]:
            ans_red_x.append(train[i][0])
            ans_red_y.append(train[i][1])

    test_green_x=[]
    test_green_y=[]
    test_red_x=[]
    test_red_y=[]
    test_blue_x=[]
    test_blue_y=[]
    wrong_x =[]
    wrong_y=[]
    for i in range(len(test_ans)):
        print(test_ans[i])
        if test_ans[i]==[1, 0, 0]:
            print("Point")
            test_green_x.append(test[i][0])
            test_green_y.append(test[i][1])
        if test_ans[i]==[0, 1, 0]:
            test_blue_x.append(test[i][0])
            test_blue_y.append(test[i][1])
        if test_ans[i]==[0, 0, 1]:
            test_red_x.append(test[i][0])
            test_red_y.append(test[i][1])
    #Clasification
    for point in all_points:
        t=np.round(nn.query(point).T,0)
        if (t==[1,0,0]).all():
            green_x.append(point[0])
            green_y.append(point[1])
        elif (t==[0,1,0]).all():
            blue_x.append(point[0])
            blue_y.append(point[1])
        elif (t==[0,0,1]).all():
            red_x.append(point[0])
            red_y.append(point[1])
        else:
            wrong_x .append(point[0])
            wrong_y.append(point[1])
    #Split train
    train_x=[]
    train_y=[]
    for tr in train:
        train_x.append(tr[0])
        train_y.append(tr[1])
    test_x=[]
    test_y=[]
    for tr in test:
        test_x.append(tr[0])
        test_y.append(tr[1])

    #Set up plot
    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    plt.title(out)
    #plt.ylim([0,0.02])
    #plt.xscale("log")
    #plt.yscale("log")
    alpha=0.12
    #Plot nn output
    plt.scatter(green_x,green_y,color='green', alpha=alpha,s=20,label="obiekt 1")
    plt.scatter(blue_x,blue_y,color='blue', alpha=alpha,s=20,label="obiekt 2")
    plt.scatter(red_x,red_y,color='red', alpha=alpha,s=20,label="obiekt 3")
    plt.scatter(wrong_x ,wrong_y,color='gold',alpha=alpha, s=20,label="bład identyfikacji")

    #Plot traing points
    plt.scatter(ans_green_x,ans_green_y,color='green',marker="X", s=20,label="punkty treningowe - 1")
    plt.scatter(ans_blue_x,ans_blue_y,color='blue',marker="X", s=20,label="punkty treningowe - 2")
    plt.scatter(ans_red_x,ans_red_y,color='red', marker="X",s=20,label="punkty treningowe - 3")
    #Plot test points
    plt.scatter(test_green_x,test_green_y,color='green',marker="D", s=20,label="punkty testowe - 1")
    plt.scatter(test_blue_x,test_blue_y,color='blue',marker="D", s=20,label="punkty testowe - 2")
    plt.scatter(test_red_x,test_red_y,color='red',marker="D", s=20,label="punkty testowe - 3")

    #plt.xlabel("Number of iterations",fontsize="xx-large")
    #plt.ylabel("Mean squared error", fontsize="xx-large")
    #ax.plot(x,y,color='purple')
    lgd=plt.legend()
    plt.savefig(out+".png",bbox_extra_artists=(lgd,), bbox_inches='tight')




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

iterable=[0,1,2,3]


list_list=list(itertools.combinations(iterable, 2))

for ll in list_list:
    #prepare data
    train_set=[]
    test_set=[]
    for k in range(len(out_train[0])):
        #set of points to train
        train_tab=[]
        for kk in ll:
            ##list kk, k element
            train_tab.append(out_train[kk][k])
        train_set.append(train_tab)
    for k in range(len(out_test[0])):
        #set of points to train
        test_tab=[]
        for kk in ll:
            ##list kk, k element
            test_tab.append(out_test[kk][k])
        test_set.append(test_tab)
    iter=0
    input_nodes = 2
    hidden_nodes = 17
    output_nodes = 3
    learningrate = 0.1
    nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes)
    while iter<2*10**3:
        #train single epoch
        #k-point to train
        for k in range(len(train_set)):
            nn.train(train_set[k],ans_train[k])
        f=nn.query
        error=MSE(f,train_set,ans_train)
        iter+=1
    plotChart(nn,train_set,test_set,ans_train,ans_test,"Granice_decyzyjne"+str(ll))
