import numpy as np 
import scipy as sp
import scipy.linalg
from functions import *


np.random.seed(10)

"""
Compute homography x' = Hx for 4-vectors. 
Alternatively, compute for any dimension...

Apply the DLT algorithm to compute the homography, as in p.91 and p.109
Hartley and Zisserman
"""
def centroid3d(array_of_points):
    """
    Compute the centroid of arbitrarily many points in a np array. 

    Return x, y, z of centroid 
    """
    assert array_of_points.shape[1] == 3, "Need to be inhomogeneous 3-vectors"
    length = array_of_points.shape[0]
    sumx = np.sum(array_of_points[:,0])     # Average x-positions
    sumy = np.sum(array_of_points[:,1])     # Average y-positions
    sumz = np.sum(array_of_points[:,2])     # Average z-positions

    return sumx/length, sumy/length, sumz/length

def translate3d(arr):
    """
    Return the array of newpoints that has the centroid at the origin.
    """
    x_c, y_c, z_c = centroid3d(arr)
    T = np.array([[1,0,0,-x_c], [0,1,0,-y_c], [0,0,1,-z_c], [0,0,0,1]])
    trans_arr = np.zeros((arr.shape[0], arr.shape[1]+1))

    for i in range(len(arr)):
        trans_arr[i] = T @ inhomogeneous2homogeneous(arr[i])

    inh_trans_arr = np.zeros(arr.shape)
    for i in range(len(arr)):
        inh_trans_arr[i] = homogeneous2Inhomogeneous(trans_arr[i])

    return inh_trans_arr

def scale3d(arr):
    """
    Fix the scale of the array of 3D points (x_i, y_i, z_i) such that \
    the average point is (1,1,1) - i.e the average displacement from \
    the origin is sqrt(3). 

    Return the scale factor. 
    """
    displacement = avgDisplacement(arr) # Average displacement from origin
    scale = np.sqrt(3) / displacement   # Find scale factor
    return scale


def normalise3D(X):
    """
    Given the set of 3D points {X_i}, find the transformation matrix T \
    that brings the centroid to the origin, and scales the points such \
    that the average displacement from the origin is sqrt(3). 

    Return the translation matrix T. 
    """
    x_c, y_c, z_c = centroid3d(X)   # Find the centroid coordinates
    newpoints = translate3d(X)      
    scale = scale3d(newpoints)      # Find the scale factor

    T = np.array([[1, 0, 0, -x_c], [0,1, 0,-y_c], [0,0, 1, -z_c], [0,0,0,1]])
    T *= scale
    T[-1][-1] = 1    # Make sure the last entry is 1...

    return T


def DLT_form_A(X1, X2):
        """
        Form the matrix A (Eq 4.3 in Hartley and Zisserman) as in the \
        DLT algorithm. 

        For n ground control points, there will be n 3 x 16 matrices A_i. \
        Use these matrices to form a 3n x 16 matrix A. 
        
        Return matrix A
        """

        assert len(X1) == len(X2), "There must be equal number of points! "


        A = np.zeros((3*len(X1), 16))
        # lists to store the data as we fill the array...
        col1  = []
        col2  = []
        col3  = []
        col4  = []
        col5  = []
        col6  = []
        col7  = []
        col8  = []
        col9  = []
        col10 = []
        col11 = []
        col12 = []
        col13 = []
        col14 = []
        col15 = []
        col16 = []

        for i in range(len(X1)):
            x1, y1, z1 = X1[i][:3]
            x2, y2, z2 = X2[i][:3]

            col1.extend((x1, 0, 0))
            col2.extend((y1, 0, 0))
            col3.extend((z1, 0, 0))
            col4.extend((1, 0, 0))
            col5.extend((0, x1, 0))
            col6.extend((0, y1, 0))
            col7.extend((0, z1, 0))
            col8.extend((0, 1, 0))
            col9.extend((0,0,x1))
            col10.extend((0,0,y1))
            col11.extend((0,0,z1))
            col12.extend((0,0,1))
            col13.extend((-x2*x1, -y2*x1, -z2*x1))
            col14.extend((-x2*y1, -y2*y1, -z2*y1))
            col15.extend((-x2*z1, -y2*z1, -z2*z1))
            col16.extend((-x2, -y2, -z2))

        A[:,0], A[:,1], A[:,2], A[:,3] = col1, col2, col3, col4
        A[:,4],A[:,5], A[:,6], A[:,7]  = col5, col6, col7, col8
        A[:,8], A[:,9], A[:,10] = col9, col10, col11 
        A[:,11], A[:,12], A[:,13] = col12, col13, col14 
        A[:,14], A[:,15] = col15, col16

        return A

def DLT(X1, X2):
    """
    Carry out Algorithm 4.2 as in Hartley and Zisserman to compute the \
    homography H that satisfies the equation relating two inhomogeneous \
    3-vectors X1 and X2, such that X2 = H X1. 

    Return the homography H 
    """

    assert X1.shape[1] == 3, "X1 and X2 must be given as inhomogeneous 3-vectors"
    # 1. Normalise the coordinates T.
    T1 = normalise3D(X1)
    T2 = normalise3D(X2) 

    X1_norm = np.zeros((X1.shape[0], 4))
    X2_norm = np.zeros((X2.shape[0], 4))

    for i in range(len(X1)):
        X1_norm[i] = T1 @ inhomogeneous2homogeneous(X1[i])
        X2_norm[i] = T2 @ inhomogeneous2homogeneous(X2[i])
    
    # form matrix A
    A = DLT_form_A(X1_norm, X2_norm)

    # solve using single value decomposition
    u, d, vt = np.linalg.svd(A)
    i = np.argmin(d)
    H = vt[i].reshape(4,4)

    # unnormalise the matrix H to obtain the correct homography

    Hnew = np.linalg.inv(T2) @ H @ T1

    return Hnew