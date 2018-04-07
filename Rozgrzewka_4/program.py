import matplotlib.pyplot as plt
import numpy as np

def getPointsInCircle(center,radius,amount):
    centerX=center[0]
    centerY=center[1]

    ansX=[]
    ansY=[]
    while len(ansX)<amount:
        randX=np.random.uniform(centerX-radius,centerX+radius)
        randY=np.random.uniform(centerY-radius,centerY+radius)
        if (centerX-randX)**2+(centerY-randY)**2<radius**2:
            ansX.append(randX)
            ansY.append(randY)
    return ansX,ansY
def getPointsInSquare(center,radius,amount):
    centerX=center[0]
    centerY=center[1]
    ansX=[]
    ansY=[]
    while len(ansX)<amount:
        randX=np.random.uniform(centerX-radius,centerX+radius)
        randY=np.random.uniform(centerY-radius,centerY+radius)
        ansX.append(randX)
        ansY.append(randY)
    return ansX,ansY

def plotChart(x,y,r_x,r_y,out):
        #Set up plot
        fig=plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        plt.grid()
        ax.scatter(x,y,color='purple')
        ax.scatter(r_x,r_y,color='red')
        c1=plt.Circle([-3,0],2,alpha=0.5)
        c2=plt.Circle([3,0],2,alpha=0.5)
        plt.xlim([-10,10])
        plt.ylim([-10,10])
        ax.add_artist(c1)
        ax.add_artist(c2)
        plt.savefig(out)

def plotPointsOfDict(x,y,r_x,r_y,data,out):
        #colors=iter(cm.Set1(np.linspace(0,1,len(data))))
        #Set up plot
        fig=plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        plt.grid()
        ax.scatter(x,y,color='purple')
        ax.scatter(r_x,r_y,color='red')
        c1=plt.Circle([-3,0],2,alpha=0.5)
        c2=plt.Circle([3,0],2,alpha=0.5)
        for red,blues in data.items():
            for b in blues:
                plt.plot([red[0],b[0]],[red[1],b[1]],'--', lw=0.3,color="black")

        plt.xlim([-10,10])
        plt.ylim([-10,10])
        ax.add_artist(c1)
        ax.add_artist(c2)
        plt.savefig(out)

def dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


x,y=getPointsInCircle([-3,0],2,100)
x1,y1=getPointsInCircle([3,0],2,100)
x+=x1
y+=y1

red_X,red_Y=getPointsInSquare([0,0],10,4)

for n in range(3):
    red_black={}
    for i in range(len(red_X)):
        red_black[(red_X[i],red_Y[i])]=[]

    for i in range(len(x)):
        dist_dict={}
        #print("RED: ")
        for red in range(len(red_X)):
            #print(red_X[red],"  ",red_Y[red])
            d=dist(red_X[red],red_Y[red],x[i],y[i])
            dist_dict[d]=[red_X[red],red_Y[red]]
        a=min(dist_dict)
        red_black[(dist_dict[a][0],dist_dict[a][1])].append([x[i],y[i]])

    plotPointsOfDict(x,y,red_X,red_Y,red_black,"k-średnie-iter{0}".format(n))

    #new red points
    new_red_x=[]
    new_red_y=[]
    for r,blues in red_black.items():
        x_n=[]
        y_n=[]
        for b in blues:
            x_n.append(b[0])
            y_n.append(b[1])
        if (len(x_n)>0 and len(y_n)>0):
            new_red_x.append(np.mean(x_n))
            new_red_y.append(np.mean(y_n))
    red_X=new_red_x
    red_Y=new_red_y
