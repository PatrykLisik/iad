''' Common functions '''
import numpy as np
import csv

#Mean squared error
def MSE(f,intput,output):
    ans=0;
    for i in range(len(output)):
        ans+=((f(intput[i]).T-output[i])**2)
    return np.sum(ans)/2

#Custom ticks to plot functions
def myticks(x,pos):
    if x <= 0:
         return "$0$"
    exponent = int(np.log10(x))
    coeff = x/10**exponent
    return r"${:2.1f} \times 10^{{ {:2d} }}$".format(coeff,exponent)
def getAllData(intput):
    reader = csv.reader(intput)
    outX=[]
    for row in reader:
        i = list(map(float,row[0].split(" ")))
        outX.append(i)
    return outX
