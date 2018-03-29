''' Common functions '''
import csv


#Mean squared error
def MSE(f,intput,output):
    ans=0;
    for i in range(len(output)):
        ans+=((f(intput[i])-output[i])**2)
    return ans/2 

def getData(intput):
    reader = csv.reader(intput)
    outX=[]
    for row in reader:
        i = list(map(float,row[0].split(" ")))
        outX.append(i)
    return outX
