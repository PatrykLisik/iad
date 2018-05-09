# How it works
## Abstract
To approximate given set of points with smaller number of points
## Algorithm
1. Choose a random point from given set
1. Find a neuron which is the closest within the meaning of chosen distance function
1. Here two approaches are possible:
    * WTA(Winner takes all) only the closest neuron is moved towards point
    * WTM(Winner takes most) winner's move multiplier is the highest and other neurons can also be moved based on theirs distance from winner in network.
        - Algorithm does not specify what function should be used to determine how to compute multipliers, but usually function is based on standard distribution
1. Repeat until end condition isn't met
    - Number of iterations
    - Given error

## Implementation

### Storing the network
Implementation covers cases of n-dimensional set of points approximated by k-dimensional network. Point in n-dimensional space can be represent as tuple
of n values. For every neuron its current position in space and its position in
network are needed to be stored. Network is represent by dictionary(hash table
in python) with single entrance as position_in_network:position_in_space.  
