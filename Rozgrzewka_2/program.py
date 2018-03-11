#!/usr/bin/env python3
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from Aprox import *
import csv
import sys

#Mean squared error
def MSE(f,intput,output):
    ans=0;
    for i in range(len(output)):
        ans+=((f(intput[i])-output[i])**2)
    return ans/len(output)

def getData(intput):
    reader = csv.reader(intput)
    outX=[]
    outY=[]
    for row in reader:
        i = list(map(float,row[0].split(";")))
        outX.append(i[0:-1])
        outY.append(i[-1])
    return outX,outY
def genName(intputFileName):
    noExt=intputFileName.split(".")[0]
    return "quality"+noExt[-1]+".png"
def plotChart(x,y,out):
        #Set up plot
        plt.figure(figsize=(10,10))
        plt.grid()
        plt.title("")
        #plt.ylim([0,0.02])
        #plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of iterations",fontsize="xx-large")
        plt.ylabel("Mean squared error", fontsize="xx-large")
        plt.plot(x,y,color='purple')
        plt.savefig(out)

intput_file = open(sys.argv[1], "r+")
PointsX,PointsY=getData(intput_file)
intput_file.close()

aprox=Aprox(len(PointsX[0]))

Error=1
ErrorY=[]
ErrorX=[]
nOfIter=0
while Error>10**(-5):
    Error=MSE(aprox,PointsX,PointsY);
    for i in range(len(PointsX)):
        aprox.updateWeigths(PointsX[i],PointsY[i])
    ErrorY.append(Error)
    ErrorX.append(nOfIter)
    """if nOfIter%20==0:
        print("Error:",Error)
        print("Iteration:",nOfIter)"""
    nOfIter+=1
#print to stdout
for i in aprox.W:
    for j in i:
        print(j)
print(aprox.w0[0])
plotChart(ErrorX,ErrorY,genName(sys.argv[1]))
