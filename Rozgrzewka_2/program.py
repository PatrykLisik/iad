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
        outX.append(i[0:-1][0])
        outY.append(i[-1])
    return outX,outY

intput_file = open(sys.argv[1], "r+")
x,y=getData(intput_file)
intput_file.close()
print("X: ",x,"\nY:",y)
#points
PointsX=x
PointsY=y

#Set up plot
plt.figure(figsize=(10,10))
plt.grid()
plt.title("")
#plt.ylim([0,0.02])
#plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of iterations",fontsize="xx-large")
plt.ylabel("Mean squared error", fontsize="xx-large")

aprox=Aprox(1)

Error=1
ErrorY=[]
ErrorX=[]
nOfIter=0
while Error>10**(-4):
    Error=MSE(aprox,PointsX,PointsY);
    for i in range(len(PointsX)):
        aprox.updateWeigths(PointsX[i],PointsY[i])
    ErrorY.append(Error)
    ErrorX.append(nOfIter)
    if nOfIter%20000==0:
        print("Error:",Error)
        print("Iteration:",nOfIter)
    nOfIter+=1;
#print to stdout
for i in aprox.W:
    print(i[0])
print(aprox.w0[0][0])

print("total number of iterations:",nOfIter)

plt.plot(ErrorX,ErrorY,color='purple')
plt.savefig("Errors1.png")
