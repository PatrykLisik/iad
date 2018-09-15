# Warm-up 2

## Before you start

It is worth mentioning that this thing is little hard. To understand it competently, you need to have at least 2 seasons of Rick and Morty watched to understand math and code without preparation.

## Abstract

Approximate n-dimensional sigmoid function to given points with gradient descent

![equation](http://mathurl.com/yd8zpy5h.png)

where

 ![equation](http://mathurl.com/yazyvsvw.png) is a point in n-dimensional space

![equation](http://mathurl.com/ycj8go8w.png) is vector of weights

![equation](http://latex.codecogs.com/gif.latex?$$w_0$$) can be considered as bias

## Solution

 Let denote error function as

![equation](http://mathurl.com/y8rd2uu9.png)

 where

 ![equation](http://mathurl.com/y8ad3puu.png) is given point

To find best weights, it is necessary to find global minimum, which is not a feasible thing to do. Decent result can be achieved by finding a local minimum.

![equation](http://mathurl.com/y7z4glto.png)

Derivative of sigmoid function

![equation](http://mathurl.com/y9lkaqq7.png)

Almost every partial derivative
![equation](http://mathurl.com/yakgneqa.png)
For ![equation](http://mathurl.com/y8zd3hj9.png) everything is working pretty similar, but there is no x at the end because
 ![equation](<http://mathurl.com/y9c26ow2.png>)

To sum up for ![equation](http://mathurl.com/y8zd3hj9.png)

![equation](http://mathurl.com/yddqkkxr.png)

The whole gradient looks like this

![equation](http://mathurl.com/y8qr9nk6.png)

## Algorithm

1.  Get points
2.  Compute gradient and multiply it by "step"
3.  Subtract gradient from weights
4.  If error is higher then target one go to 1.

## Visualization how it works

Script solv.py generates chart with 5 given points and curves after some numbers of learning epoch. After every iteration curve sticks closer and closer to points.


![alt text](https://github.com/PatrykLisik/iad/blob/master/Perceptron/show_learn.png "Chart 1")
