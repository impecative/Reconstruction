from __future__ import division, print_function # python 2 compatibility.
from functions import *
import numpy as np
from numpy.linalg import inv, multi_dot, solve
import scipy.optimize

# step 1 
def getTransformationMatrices(coordinate1, coordinate2):
    """Define transformation matrices T1 and T2 that take coordinates
    (x1,y1) and (x2, y2) back to the origin, respectively."""
    x1, y1 = coordinate1[:2]
    x2, y2 = coordinate2[:2]

    x1Translation, y1Translation = centroid(np.array([[x1, y1]]))
    x2Translation, y2Translation = centroid(np.array([[x2, y2]]))

    T1 = np.array([[1, 0, -x1Translation], [0, 1, -y1Translation], [0, 0, 1]])
    T2 = np.array([[1, 0, -x2Translation], [0, 1, -y2Translation], [0, 0, 1]])

    return T1, T2

# example... 
# T1, T2 = getTransformationMatrices([1, 2], [3, 5])
# print(T1)
# print(T2)

# step 2
def translateFundamentalMatrix(F, T1, T2):
    """Replace matrix F with transpose(inv(T2)) F inv(T1)."""
    T2inv = inv(T2)
    T1inv = inv(T1)
    return multi_dot([T2inv.T, F, T1inv])

# # example (verified by hand, result as expected! )
# T1, T2 = getTransformationMatrices([1, 2], [3, 5]) 
# F  = np.array([[3,4, 6],[5,3, 6], [3,5, 1]])
# newF = translateFundamentalMatrix(F, T1, T2)
# print("Original fundamental matrix F =\n", F)
# print("T1 translation matrix is \n", T1)
# print("T2 translation matrix is \n", T2)
# print("The new F = transpose(inv(T2)) dot F dot inv(T1) is:\n")
# print(newF)

# Step 3
def findEpipoles(F):
    """Find the right and left epipoles e1 and e2 such that 
    e'.T F = 0 and F e = 0. Return normalsed epipoles..."""
    F_11, F_12, F_13, F_21, F_22, F_23, F_31, F_32, F_33 = F.ravel()

    # first require e2.T dot F = 0  # force solution not (0,0,0) by setting e3 = 1 
    # ... This setting is "undone" by normalisation anyway...
    a = np.array([[F_11, F_21, F_31], [F_12, F_22, F_32], [F_13, F_23, F_33], [0, 0, 1]])
    b = np.array([0, 0, 0, 1])

    e2 = np.linalg.lstsq(a,b, rcond=None)[0]


    # next require F dot e1 = 0
    a = np.array([[F_11, F_12, F_13], [F_21, F_22, F_23], [F_31, F_32, F_33], [0,0,1]])
    b = np.array([0, 0, 0, 1])

    e1 = np.linalg.lstsq(a,b, rcond=None)[0]

    assert np.isclose(np.linalg.norm(np.matmul(e2.T, F)), 0), "First epipole DOESN'T satisfy the epipolar constraint"
    assert np.isclose(np.linalg.norm(np.matmul(F, e1)), 0), "Second epipole DOESN'T satisfy the epipolar constraint" 

    # print("e1 is ", e1)
    # print("e2 is ", e2)

    # normalise the epipoles such that (e11**2 + e12**2 = 1, and e21**2 + e22**2 = 1)
    scaleFactor1 = 1/(e1[0]**2 + e1[1]**2)**.5
    scaleFactor2 = 1/(e2[0]**2 + e2[1]**2)**.5

    # print("The scale factors for the epipoles are: {} and {}".format(scaleFactor1, scaleFactor2))

    newe1, newe2 = scaleFactor1*e1, scaleFactor2*e2

    return newe1, newe2

# # TEST, FN not working! 
# F = np.array([[1,2,3], [0,6,4], [0,0,0]])
# x1, x2 = [1,2], [3,4]
# T1, T2 = getTransformationMatrices(x1, x2)
# newF = translateFundamentalMatrix(F, T1, T2)
# newF = np.array([[4, -3, -4], [-3, 2, 3], [-4, 3, 4]])


# print("The fundamental matrix F=\n", F)
# print("The Transformation matrices are \nT1 = {} \n T2 = {}".format(T1, T2))
# print("The New F matrix is \nF = {} ".format(newF))
# e1, e2 = findEpipoles(newF)
# print("e = ", e1)
# print("e' = ", e2)


# Step 4
def getRotationMatrices(e1, e2):
    """Given two epipoles e1 = (e11, e12, e13)^T and 
    e2 = (e21, e22, e23)^T, return rotation matrices R
    and R' such that Re1 = (1, 0, e13)^t and Re2 = (1,0,e23)^T)."""
    e11, e12, e13 = e1.ravel()
    e21, e22, e23 = e2.ravel()

    R1 = np.array([[e11, e12, 0], [-e12, e11, 0], [0, 0, 1]])
    R2 = np.array([[e21, e22, 0], [-e22, e21, 0], [0, 0, 1]])

    return R1, R2

# Step 5
def rotateFundamentalMatrix(F, R1, R2):
    """Replace matrix F with (R2 F R1.T)"""
    return multi_dot([R2, F, R1.T])

# Step 6 and 7, TBC
def g(t, f1, f2, a, b, c, d):
    term1 = t*((a*t+b)**2 + f2**2*(c*t + d)**2)**2
    term2 = (a*d - b*c)*(1 + f1**2 * t**2)**2 * (a*t+b)*(c*t+d)

    return term1 - term2

def formPolynomial(e1, e2, F):
    """Given two epipoles e1=(e11,e12,e13)^T and 
    e2=(e21, e22, e23)^T and fundamental matrix F, 
    form polynomial g(t)... """
    # extract parameters from epipoles and matrix F
    f = e1[2]    # f
    g = e2[2]    # f' in literature
    F_11, F_12, F_13, F_21, F_22, F_23, F_31, F_32, F_33 = F.ravel()    # could be optimised by only getting necessary values.
    a, b, c, d = F_22, F_23, F_32, F_33

    return a, b, c, d, f, g

def solvePolynomial(a, b, c, d, f, g):
    """Solve the polynomial (12.7) and find its roots"""
    # first get the polynomial coefficients so we can find roots
    t6 = (a*b*c**2*f**4 - a**2*c*d*f**4)  # checked
    t5 = (a**4 + 2*a**2*c**2*g**2 - a**2*d**2*f**4 + b**2*c**2*f**4 + c**4*g**4) # checked
    t4 = (4*a**3*b - 2*a**2*c*d*f**2 + 4*a**2*c*d*g**2 + 2*a*b*c**2*f**2 + 4*a*b*c**2*g**2 - a*b*d**2*f**4 + b**2*c*d*f**4 + 4*c**3*d*g**4) # checked
    t3 = (6*a**2*b**2 - 2*a**2*d**2*f**2 + 2*a**2*d**2*g**2 + 8*a*b*c*d*g**2 + 2*b**2*c**2*f**2 + 2*b**2*c**2*g**2 + 6*c**2*d**2*g**4)  # checked
    t2 = (-a**2*c*d + 4*a*b**3 + a*b*c**2 - 2*a*b*d**2*f**2 + 4*a*b*d**2*g**2 + 2*b**2*c*d*f**2 + 4*b**2*c*d*g**2 + 4*c*d**3*g**4) # checked
    t1 =  (- a**2*d**2 + b**4 + b**2*c**2 + 2*b**2*d**2*g**2 + d**4*g**4) # checked
    t0 =  (- a*b*d**2 + b**2*c*d) # checked

    coefs = [t6, t5, t4, t3, t2, t1, t0]

    # find the roots of the sixth order polynomial
    roots = np.roots(coefs)

    # check they are actually roots...

    counter = 0
    for t in roots:
        val = np.polyval(coefs, t.real)
        if np.isclose(val, 0):
            counter += 1

    assert counter > 0, "No roots were found..."

    return roots

# Step 8
def costFunction(t, a, b, c, d, f, g):

    return (t**2/(1+f**2*t**2)) + (c*t+d)**2/((a*t+b)**2 + g**2*(c*t+d)**2) 

def evaluateCostFunction(roots, a, b, c, d, f, g):
    """Evaluate the cost function (12.5) for input array/list np.array([t1,t2, ...]).\n
    Select the value of t for which the cost function is the smallest."""

    tmin = 999999
    minCostFn = 99999999
    
    for t in roots:
        costFn = costFunction(t.real, a, b, c, d, f, g)
        if costFn < minCostFn:
            minCostFn = costFn
            tmin = t.real
        else:
            pass
    
    assert tmin != 999999, "No minimum value of t has been found..."
    assert minCostFn != 99999999, "The cost function has not been minimised..."

    # also find the value of cost function as t=infty, corresponding to 
    # an epipolar line fx=1 in the first image .

    inftyCostFn = (1/f**2) + (c**2/(a**2 + g**2*c**2))

    assert inftyCostFn > minCostFn, "The epipolar line in the first image is fx=1, tmin=infty"

    return tmin

# Step 9
def findModelPoints(tmin, a, b, c, d, f, g ):
    """Find the model points x and x' that fit the epipolar constrant x'^T F x = 0\n
    Return the coordinates x1,x2"""

    l1 = np.array([tmin*f, 1, -tmin])
    l2 = np.array([-g*(c*tmin+d), a*tmin+b, c*tmin+d])

    x1 = np.array([-l1[0]*l1[2], -l1[1]*l1[2], l1[0]**2 + l1[1]**2])
    x2 = np.array([-l2[0]*l2[2], -l2[1]*l2[2], l2[0]**2 + l2[1]**2])

    return x1, x2

# Step 10
def findOriginalCoordinates(R1, R2, T1, T2, x1, x2):
    """Transfer the normalised coordinates back to the original coordinates using the formulas:\n
    x1 = T1^-1 R1^T x1; x2 = T2^-1 R2^T x2"""

    newx1 = np.linalg.multi_dot([inv(T1), R1.T, x1])
    newx2 = np.linalg.multi_dot([inv(T2), R2.T, x2])

    return newx1, newx2

# Step 11; Triangulation...
# Homogeneous method

def formA(imgcoord1, imgcoord2, P1, P2):
    """As in S12.2, form matrix A to solve set of equations. """
    # extract the image coordinates
    x1, y1 = imgcoord1[:2]     # do we have to have these homogeneous coordinates or inhomogeneous coords? 
    x2, y2 = imgcoord2[:2]

    A = np.array([[x1*P1[2] - P1[0]], [y1*P1[2] - P1[1]], [x2*P2[2] - P2[0]], [y2*P2[2] - P2[1]]])

    # fix the formatting
    newA = np.array([A[0][0], A[1][0], A[2][0], A[3][0]])

    return newA


def find3DPoint(A):
    """Solve the set of linear equations AX = 0 for X
    
    SVD of A = UDV \n
    X is the last column of V"""

    u, d, v = np.linalg.svd(A)
    D = np.diag(d)

    X = v[:,-1]  # in homogeneous coordinates

    return v[:,-1]

def homogeneous2Inhomogenous(X):
    """Convert homogeneous coordinate [x1, x2, x3, ..., xn] to \ninhomogeneous coordinate
    [x1/xn, x2/xn, ..., xn-1/xn]"""
    divider = X[-1]
    # print(X[len(X)-2])

    newcoord = []

    for i in range(len(X)-1):
        newcoord.append(X[i]/divider)

    return newcoord

def triangulate(imgcoord1, imgcoord2, P1, P2):
    '''Triangulate the coordinates in each image back to a 3D point X, using their 
    projection matrices P1 and P2 respectively...'''
    # first form the matrix A
    A = formA(imgcoord1, imgcoord2, P1, P2)
    
    # find point X in homogeneous coordinates
    X = find3DPoint(A)

    # modify back to inhomogeneous (3-vector) coordinate

    return homogeneous2Inhomogenous(X)

# example:
# imgcoord1 = (1,2,1)
# imgcoord2 = (3, 4, 1)
# P1 = np.array([[1,4,2,1], [2,3,1,4], [1,2,3,4]])
# P2 = np.array([[1,2,3,4], [4,3,2,1], [1,2,3,4]])

# A = formA(imgcoord1, imgcoord2, P1, P2)

# X = find3DPoint(A)
# print("In homogeneous coordinates: ", X)
# print("In Real world coordinates point is at", homogeneous2Inhomogenous(X))



def main():
    # example data

    # need to provide an already computed fundamental matrix F
    # and a pair (or multiple) of corresponding points x1, x2
    F = np.array([[1,2,3], [0,6,4], [0,0,0]])
    x1, x2 = [1,2], [3,4]

    P1 = np.array([[1,4,2,1], [2,3,1,4], [1,2,3,4]])    # a random camera matrix for image 1
    P2 = np.array([[1,2,3,4], [4,3,2,1], [1,2,3,4]])    # a random camera matrix for image 2


    # compute the transformation matrices and modify the fundamental matrix
    T1, T2 = getTransformationMatrices(x1, x2)
    newF = translateFundamentalMatrix(F, T1, T2)   # newF corresponds to translated coordinates
    # newF = np.array([[4, -3, -4], [-3, 2, 3], [-4, 3, 4]])

    # check everything working correctly...
    # print("The fundamental matrix F=\n", F)
    # print("The Transformation matrices are \nT1 = {} \n T2 = {}".format(T1, T2))
    # print("The New F matrix is \nF = {} ".format(newF))


    # compute the left and right epipoles...
    e1, e2 = findEpipoles(newF)

    # form the rotation matrices and modify the fundamental matrix again
    R1, R2 = getRotationMatrices(e1, e2)
    newF = rotateFundamentalMatrix(newF, R1, R2)

    # get constants and form polynomial (12.7)
    a,b,c,d,f,g = formPolynomial(e1, e2, newF)

    # find the roots of the polynomial and check find the value of t that
    # minimises the cost function
    roots = solvePolynomial(a,b,c,d,f,g)
    tmin = evaluateCostFunction(roots, a, b, c, d, f, g)

    # find the optimal translated points
    x1, x2 = findModelPoints(tmin, a, b, c, d, f, g)

    # transfer back to the original coordinates
    newx1, newx2 = findOriginalCoordinates(R1, R2, T1, T2, x1, x2)

    # now triangulate the points newx1, newx2 back to a 3D point X
    X = triangulate(newx1, newx2, P1, P2)

    print("The 3D point has been found: ")
    print(X)




if __name__ == "__main__":
    main()








