from __future__ import division, print_function
import numpy as np
import scipy as sp

def skew(V):
    '''Form a skew symmetric matrix from 3-vector V'''
    v1, v2, v3 = V.ravel()
    return np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

def inhomogeneous2homogeneous(X):
    '''Convert the inhomogeneous coordinate X to homogeneous coordinates...'''

    newcoord = []

    for i in X:
        newcoord.append(float(i))

    newcoord.append(1.)

    return newcoord

def decomposeCameraMtx(p):
    '''Decompose camera matrix P = [M | -MC] = K[R|-RC].
    \nReturn the calibration matrix, rotation matrix and camera centre C.'''

    assert p.shape == (3,4), "camera projection matrix is the wrong shape! "

    # first obtain camera centre...
    # SVD of P, but needs square matrix so add a row of zeros.
    P = np.r_[p, np.zeros((1,4))]
    u, d, vt = np.linalg.svd(P)
    assert np.allclose(u @ np.diag(d) @ vt, P), "SVD of P has not worked correctly..."
    assert np.argmin(d) == len(d)-1, "The diagonals are not in decreasing size order..."
    # Then the camera centre is the last column of V
    x = vt[-1][0]/vt[-1][-1]     
    y = vt[-1][1]/vt[-1][-1]
    z = vt[-1][2]/vt[-1][-1]
    c = np.array([x,y,z])       # camera centre

    # now we need to find M and K,R from M = KR.
    M = p[:3,:3]
    # assert np.allclose(-M@c, p[:,-1]), "-MC is not equal to last column of p for some reason..."

    # find K, R from RQ-decomposition of M
    K, R = scipy.linalg.rq(M)

    # remove ambiguity in the RQ decomposition by making diagonals of K positive.
    T = np.diag(np.sign(np.diag(K)))
    assert np.allclose(K @ T @ T @ R, K @ R), "The transformation T is not its own inverse..."
    K = K @ T
    R = T @ R 

    # ensure that the last diagonal of K is equal to 1
    scale = 1/K[2,2]
    K = scale * K

    return K, R, c

def unit(vec):
    """Return the unit vector of vector."""
    return vec/np.linalg.norm(vec)
    
