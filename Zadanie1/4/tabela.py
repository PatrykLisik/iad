#!/usr/bin/env python3
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from NeutralNetwork import NeutralNetwork
from Functions import MSE,getData
import numpy as np
import csv

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

#HeadLines
results=[]
results.append(["Learning rate","Momentum","MSE_train","MSE_test","STD"])

for k in range(100):
    input_nodes = 1
    hidden_nodes = np.random.randint(1,20)
    output_nodes = 1
    learningrate = np.random.uniform(0,0.25)
    bias=1 #on
    momentum=np.random.uniform(0,0.0005)
    activation_function_output=lambda x: x
    dactivation_function_output=lambda x: 1
    nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes,
                        learningrate,bias,momentum,activation_function_output,
                        dactivation_function_output)

    i=0
    error_test=10
    error_train=10
    std="???"
    current_tuple=[learningrate,momentum]
    while i<1000:
        f=nn.query
        error_train=MSE(f,train_input_list,train_target_list)
        error_test=MSE(f,test_input_list,test_target_list)
        for j in range(len(train_input_list)):
            nn.train(train_input_list[j],train_target_list[j])
        i+=1
    current_tuple.append(error_train)
    current_tuple.append(error_test)
    current_tuple.append(std)
    results.append(current_tuple)
    print(current_tuple)

with open("TabelaNa4plik-{0}.csv".format(str(sys.argv[1])), "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
