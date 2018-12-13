from __future__ import division, print_function   # for python 2 compatibility.
import numpy as np
import scipy
import time
from functions import *
from a_12_1 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
recognizer = cv.face.LBPHFaceRecognizer_create()

# compute the fundamental matrix F from n>= 8 point matches {x_i, x'_i}
# first normalise the coordinates

def getTransformMtx(arr_of_points):
    '''Find the transformation that normalises a group of homogeneous coordinates'''
    x_c, y_c = centroid(arr_of_points)

    sumSquares = np.sum((arr_of_points[:,0]-x_c)**2 + (arr_of_points[:,1]-y_c)**2)

    s = np.sqrt(sumSquares/(2*len(arr_of_points)))

    T = np.array([[1/s, 0, -1/s * x_c], [0, 1/s, -1/s * y_c], [0,0,1]])

    # print(T)

    return T

# # test
# points = np.array([[4,2,1], [7,4,1], [1,3,1]])

# T = getTransformMtx(points)

# print(T)

# find the fundamental matrix F corresponding to normalised image coordinates 

def formMatrixA(arr_of_imgpoints1, arr_of_imgpoints2):
    '''Form the matrix A as in (11.3, p279).'''

    assert len(arr_of_imgpoints1) == len(arr_of_imgpoints2), "The array of image points must be matching and the same length"

    A = np.zeros((len(arr_of_imgpoints1), 9))
    A[:,-1] = 1    # set the last column to all 1s

    # lists to store values as we work them out... 

    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []

    for i in range(len(arr_of_imgpoints1)):
        x1, y1 = arr_of_imgpoints1[i][:2]   # x, y
        x2, y2 = arr_of_imgpoints2[i][:2]   # x', y'

        col1.append(x2*x1)
        col2.append(x2*y1)
        col3.append(x2)
        col4.append(y2*x1)
        col5.append(y2*y1)
        col6.append(y2)
        col7.append(x1)
        col8.append(x1)
    
    A[:,0], A[:,1], A[:,2], A[:,3], A[:,4],A[:,5], A[:,6], A[:,7]  = col1, col2, col3, col4, col5, col6, col7, col8

    return A

# TESTING - It works...
# testdata1 = np.random.randint(0, 10, size=(10, 3))
# testdata2 = np.random.randint(0, 10, size=(10, 3))
# testdata1[:,2] = 1
# testdata2[:,2] = 1 

# print("first line of test data is ", testdata1[0], testdata2[0])

# print("\n the matrix A is ")
# print(formMatrixA(testdata1, testdata2))

def solveFundamentalMatrix(A):
    '''For set of linear equations in matrix A, find f to solve Af=0. \n
    Return F, the matrix form of 9-vector f. '''

    # find SVD of A

    u, d, v = np.linalg.svd(A)

    # least squares solution for f is last column of V
    f = v[:,-1]

    F = f.reshape(3,3)

    # now we need to constrain that det(F)=0, need unique solution! 

    # take SVD of F
    u, d, v = np.linalg.svd(F)

    # check d = diag(r,s,t), with r >= s >= t. 
    assert d[0] >= d[1], "The SVD of F has produced a D = diag(r,s,t) where the contraints r >= s >= t have NOT been met..."
    assert d[1] >= d[2], "The SVD of F has produced a D = diag(r,s,t) where the contraints r >= s >= t have NOT been met..."

    # if this criteria is met then the minimised F = U diag(r,s,0) V^T
    D = np.diag([d[0], d[1], 0])

    newF = u @ D @ v

    return newF

def getOriginalFundamentalMatrix(F, T1, T2):
    '''Denormalise the fundamental matrix and replace with F = T2.T F T1 \n
    This new fundamental matrix corresponds to the original matching coordinates x1, x2.'''

    return T2.T @ F @ T1

def findCameras(F):
    '''Given fundamental matrix F, compute the cameras P = [I|0] and P'=[skew(e')F|e'] '''

    # we can define P straight away.
    P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

    # Now we need to compute the epipole e' which satisfies e'.T F = 0. 
    F_11, F_12, F_13, F_21, F_22, F_23, F_31, F_32, F_33 = F.ravel()
    a = np.array([[F_11, F_21, F_31], [F_12, F_22, F_32], [F_13, F_23, F_33], [0, 0, 1]])
    b = np.array([0, 0, 0, 1])

    e2 = np.linalg.lstsq(a,b, rcond=None)[0]    # this is e' in literature.

    # print("Epipole e' is ",e2) 

    e2_skew = skew(e2)
    # print("The skew matrix e2 is ", e2_skew)

    leftMatrix = e2_skew @ F 
    P2 = np.c_[leftMatrix, e2]

    return P1, P2

def cameraMatrices(img1_points, img2_points):
    img1coords = img1_points
    img2coords = img2_points

    # define the transformation matrices that normalse the coordinates...
    T1 = getTransformMtx(img1coords)
    T2 = getTransformMtx(img2coords)

    # normalise the coordinates
    normImg1coords = []
    normImg2coords = []

    for i in range(len(img1coords)):
        normImg1coords.append(T1 @ img1coords[i])
        normImg2coords.append(T2 @ img2coords[i])

    # for set of linear equations A from image coordinates
    A = formMatrixA(normImg1coords, normImg2coords)

    # solve set of equations to find fundamental matrix corresponding to TRANSFORMED coordinates
    newF = solveFundamentalMatrix(A)

    F = getOriginalFundamentalMatrix(newF, T1, T2)

    p1, p2 = findCameras(F)

    return p1, p2

def estimate3DPoints(img1_points, img2_points):
    img1coords = img1_points
    img2coords = img2_points

    # define the transformation matrices that normalse the coordinates...
    T1 = getTransformMtx(img1coords)
    T2 = getTransformMtx(img2coords)

    # normalise the coordinates
    normImg1coords = []
    normImg2coords = []

    for i in range(len(img1coords)):
        normImg1coords.append(T1 @ img1coords[i])
        normImg2coords.append(T2 @ img2coords[i])

    # for set of linear equations A from image coordinates
    A = formMatrixA(normImg1coords, normImg2coords)

    # solve set of equations to find fundamental matrix corresponding to TRANSFORMED coordinates
    newF = solveFundamentalMatrix(A)

    F = getOriginalFundamentalMatrix(newF, T1, T2)

    p1, p2 = findCameras(F)

    ########################################################################
    # now triangulate the points! 

    # set up array to store the 3D coordinates...
    points3D = np.zeros((len(img1coords), 3))

    for i in range(len(img1coords)):
        x1, x2 = img1coords[i], img2coords[i]

        T1, T2 = getTransformationMatrices(x1, x2)
        newF = translateFundamentalMatrix(F, T1, T2)   # newF corresponds to translated coordinates

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
        X = triangulate(newx1, newx2, p1, p2)

        # now store this 3D point
        points3D[i] = X
    
    return points3D






def main():
    # the normalise 8-point algorithm for the computation of fundamental matrix F from a series of 
    # point correspondences x_i to x'_i 

    # example data
    img1coords = np.random.randint(0, 100, size=(200, 3))

    img2coords = np.zeros(img1coords.shape)

    for i in range(img1coords.shape[0]):
        for j in range(img1coords.shape[1]):
            img2coords[i][j] = img1coords[i][j] + np.random.randint(-5, 5)   # add some small displacement for matching images

    img1coords[:,-1] = 1
    img2coords[:,-1] = 1


    T1 = getTransformMtx(img1coords)    # transformation matrix for image 1
    T2 = getTransformMtx(img2coords)    # transformation matrix for image 2

    normImg1coords = []
    normImg2coords = []

    for i in range(len(img1coords)):
        normImg1coords.append(T1 @ img1coords[i])
        normImg2coords.append(T2 @ img2coords[i])

    # for set of linear equations A from image coordinates
    A = formMatrixA(normImg1coords, normImg2coords)

    # solve set of equations to find fundamental matrix corresponding to TRANSFORMED coordinates
    newF = solveFundamentalMatrix(A)

    F = getOriginalFundamentalMatrix(newF, T1, T2)

    # print(F)

    # OpenCV find fundamental matrix...
    cvF = cv.findFundamentalMat(img1coords, img2coords)[0]

    # print("OpenCV finds the fundmamental matrix to be: ")
    P1, P2 = findCameras(cvF)
    K, R, C = decomposeCameraMtx(P2)
    print(P2)

    p1, p2 = findCameras(F)

    # print("The cameras corresponding to F are: ")
    # print("P  = ", p1)
    print("P' = ", p2)


def actualData():
    # import the image coordinates...
    outfile = np.load("imgpoints.npz")
    img1coords = outfile["imgpoints1"]
    img2coords = outfile["imgpoints2"]

    # define the transformation matrices that normalse the coordinates...
    T1 = getTransformMtx(img1coords)
    T2 = getTransformMtx(img2coords)

    # normalise the coordinates
    normImg1coords = []
    normImg2coords = []

    for i in range(len(img1coords)):
        normImg1coords.append(T1 @ img1coords[i])
        normImg2coords.append(T2 @ img2coords[i])

    # for set of linear equations A from image coordinates
    A = formMatrixA(normImg1coords, normImg2coords)

    # solve set of equations to find fundamental matrix corresponding to TRANSFORMED coordinates
    newF = solveFundamentalMatrix(A)

    F = getOriginalFundamentalMatrix(newF, T1, T2)

    # print(F)

    p1, p2 = findCameras(F)

    # print("The cameras corresponding to F are: ")
    # print("P  = ", p1)
    # print("P' = ", p2)

    ########################################################################
    # now triangulate the points! 

    # set up array to store the 3D coordinates...
    points3D = np.zeros((len(img1coords), 3))

    for i in range(len(img1coords)):
        x1, x2 = img1coords[i], img2coords[i]

        T1, T2 = getTransformationMatrices(x1, x2)
        newF = translateFundamentalMatrix(F, T1, T2)   # newF corresponds to translated coordinates

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
        X = triangulate(newx1, newx2, p1, p2)

        # now store this 3D point
        points3D[i] = X

    print(points3D)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = points3D[:,0], points3D[:,1], points3D[:,2]
    ax.scatter(xs, ys, zs)
    ax.scatter(0,0,0, c="r", label="Centre of first camera")

    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")

    plt.show()






    









if __name__ == "__main__":
    main()
    # actualData()




