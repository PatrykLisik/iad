# Warm-up 2
## Before you start

It is worth mentioning that this thing is little hard and to understand competently, you need to have at least 2 seasons of Rick and Morty watched to understand it without preparation. 
## Abstract
Approximate n-dimensional sigmoid function to given points with gradient descent

![equation](http://mathurl.com/yd8zpy5h.png)

where

 ![equation](http://mathurl.com/yazyvsvw.png) is a point in n-dimensional space


![equation](http://mathurl.com/ycj8go8w.png) is vector of weights

![equation](http://latex.codecogs.com/gif.latex?$$w_0$$) can be considered as bias

## Solution
 Let denote error function as

![equation](http://mathurl.com/y9cjynwn.png)

 where

 ![equation](http://mathurl.com/y8ad3puu.png) is given point

To find best weights, it is necessary to find global minimum, which is not a feasible thing to do. Decent result can be achieved by finding a local minimum.

![equation](http://mathurl.com/ycw4l6ok.png)

![equation](http://mathurl.com/yakgneqa.png)
For ![equation](http://mathurl.com/y8zd3hj9.png) everything is working pretty similar, but there is no x at the end because ![equation](http://mathurl.com/y9c26ow2 .png)

To sum up for ![equation](http://mathurl.com/y8zd3hj9.png)

![equation](http://mathurl.com/yddqkkxr.png)

The whole gradient looks like this

![equation](http://mathurl.com/y8qr9nk6.png)
