#!/usr/bin/env python3
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from NeutralNetwork import NeutralNetwork
from Functions import MSE,round
import numpy as np
import itertools
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

def recognitionPerc(input,ans,nn):
    good_ans=0
    length=len(input)
    for test in range(length):
        t=np.round(nn.query(input[test]).T,0)
        good_ans+=clas_test(ans[test],t)
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


iterable=[0,1,2,3]


results=[]
headline=("Ilość neuronów","Numery kolumn z danymi","Średni błąd - zbiór treningowy","Średni błąd - zbiór testowy",
                "Średnie odchylenie standarowe błąd - zbiór treningowy","Średnie odchylenie standarowe - zbiór testowy")
results.append(headline)
print(headline)


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

        for hnodes in range(1,18,4):
            input_nodes = input_neuron_number
            hidden_nodes = hnodes
            output_nodes = 3
            learningrate = 0.1
            nn = NeutralNetwork(input_neuron_number, hidden_nodes, output_nodes)

            percent_test_tab=[]
            percent_train_tab=[]
            for loop in range(100):
                iter=0
                while iter<3*10**3:
                    #train single epoch
                    #k-point to train
                    for k in range(len(train_set)):
                        nn.train(train_set[k],ans_train[k])
                    f=nn.query
                    iter+=1
                #Compute percentage
                percent_train=recognitionPerc(train_set,ans_train,nn)
                percent_test=recognitionPerc(test_set,ans_test,nn)
                #Add to list
                percent_test_tab.append(percent_test)
                percent_train_tab.append(percent_train)

            #Means
            error_test_mean=np.mean(percent_test_tab)
            error_train_mean=np.mean(percent_train_tab)
            #Std by default is population std ddof=1 is sample std
            error_test_std=np.std(percent_test_tab,ddof=1)
            error_train_std=np.std(percent_train_tab,ddof=1)

            #Make tuple and append to result
            tuple_to_append=(hidden_nodes,error_train_mean,
                             error_test_mean,error_train_std,error_test_std)
            tuple_to_append=tuple(map(round,tuple_to_append))
            final_tuple=([tuple_to_append[0]]+list(ll)+list(tuple_to_append[1:]))
            results.append(final_tuple)
            print(final_tuple)


with open("TabelaNa5plik.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
