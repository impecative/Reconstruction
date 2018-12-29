from __future__ import division, print_function
import functions as fn
import numpy as np
import scipy as sp


### Given n>=5 ground truths and their images in both cameras, compute the direct
### reconstruction that this corresponds to! 

## try also 10.4.3 : direct metric reconstruction using the IAC. Extract K from F,
## Then metric reconstruction of the scene may be computed by using the essential
## matrix as in section 9.6. Four solutions - only one physical, test with depth. 


# Ground control method: 
# for n >= 5 ground control points {X_Ei}, which correspond to {X_i} in the 
# projective reconstruction. 
# Compute the homography such that X_Ei = H X_i, using DLT algorithm. 
# Then, the metric reconstruction is:
# P_M = P inv(H)   ;   P'_M = P' inv(H)   ;   X_Mi = H X_i

# need to link which ground control points are which reconstructed points...

def ground_truth_reconstruction(P1, P2, X_Ei, X_i, all_projected_points):
    """
    Using n>=5 ground control points, compute the metric reconstruction \
    for all the image points. 

    Parameters
    ___________
    P1 : 3 x 4 camera matrix 1
    P2 : 3 x 4 camera matrix 2
    X_Ei : "true" Euclidean coordinates of known ground control points
    X_i : Ground control points in the projective reconstruction
    all_projected_points : Coordinates of all of the reconstructed points

    Return the metric reconstruction of the cameras, P1, P2, and all the \
    points. 

    """
    
    
    H = fn.DLT(X_i, X_Ei)

    P1 = P1 @ np.linalg.inv(H)
    P2 = P2 @ np.linalg.inv(H)

    newpoints = np.zeros(all_projected_points.shape)

    for i in range(len(newpoints)):
        newpoints[i] = H @ all_projected_points[i]

    
    return P1, P2, newpoints
