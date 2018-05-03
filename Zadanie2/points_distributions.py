import numpy as np
import random
from math import pi, cos, sin


def cirlce_dist(center, radius, amount):
    centerX = center[0]
    centerY = center[1]

    ans = []
    while len(ans) < amount:
        randX = np.random.uniform(centerX - radius, centerX + radius)
        randY = np.random.uniform(centerY - radius, centerY + radius)
        if (centerX - randX)**2 + (centerY - randY)**2 < radius**2:
            ans.append([randX, randY])
    return ans


def point_on_triangle(pt1, pt2, pt3):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    s, t = sorted([random.random(), random.random()])
    return (s * pt1[0] + (t - s) * pt2[0] + (1 - t) * pt3[0],
            s * pt1[1] + (t - s) * pt2[1] + (1 - t) * pt3[1])


def triangle_dist(p1, p2, p3, amount):
    return [point_on_triangle(p1, p2, p3) for _ in range(amount)]


def point_on_circumference(h, k, r):
    theta = random.random() * 2 * pi
    return h + cos(theta) * r, k + sin(theta) * r


def circumference_dist(center, radius, amount):
    return [point_on_circumference(center[0], center[1], radius)
            for _ in range(amount)]
