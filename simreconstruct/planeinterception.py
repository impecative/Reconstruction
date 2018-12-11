from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random

def definePlane(A,B,C):
    '''Given three points A, B, C, find the plane they all lie coplanar on...'''

    normal = np.cross((C-A), (B-A))
    d = np.dot(normal, A)
    a,b,c = normal

    # equation of the plane is ax + by + cz = d

    return a, b, c, d

w, h, f = 23.5, 15.6, 50e-3

def findInterception(A, B, C, point3D, origin=np.array([0,0,0])):
    a, b, c, d = definePlane(A, B, C)
    n = np.array([a,b,c])
    D = point3D
    E = origin

    t = np.linalg.multi_dot([n, (A-D)])/np.linalg.multi_dot([n, (E-D)])

    # point of intersection is 
    x,y, z = D + (E-D)*t

    return x,y,z

BL = np.array([-w/2, -h/2, f])      # bottom left point
TL = np.array([-w/2, h/2, f])       # top left point
TR = np.array([w/2, h/2, f])        # top right point
BR = np.array([w/2, -h/2, f])       # bottom right point

x,y,z = findInterception(BL, TL, TR, np.array([0,0,100]))

print("The interception point is: ", x,y,z)