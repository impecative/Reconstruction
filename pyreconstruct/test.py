from __future__ import division, print_function  # Python 2 compatibility
import numpy as np
import scipy as sp
import scipy.linalg
import random
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import sys, os
random.seed(10)  # this should remain constant for testing, remove for true random distribution

# __author__ = "Alex Elliott"
# __version__ = "0.02"

def skew(V):
    '''Form a skew symmetric matrix from 3-vector V'''
    v1, v2, v3 = V.ravel()
    return np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

def inhomogeneous2homogeneous(X):
    '''Convert the inhomogeneous coordinate X to homogeneous coordinates...'''

    newcoord = np.zeros(len(X)+1)

    for i in range(len(X)):
        newcoord[i] = float(X[i])

    newcoord[-1] = 1.

    return newcoord

def decomposeCameraMtx(p):
    '''Decompose camera matrix P = [M | -MC] = K[R|-RC].
    \nReturn the calibration matrix, rotation matrix and camera centre C.'''

    assert p.shape == (3, 4), "camera projection matrix is the wrong shape! "

    # first obtain camera centre...
    # SVD of P, but needs square matrix so add a row of zeros.
    P = np.r_[p, np.zeros((1, 4))]
    _, d, vt = np.linalg.svd(P)

    # assert np.allclose(u @ np.diag(d) @ vt, P), "SVD of P has not worked correctly..."
    assert np.argmin(d) == len(d) - 1, "The diagonals are not in decreasing size order..."
    # Then the camera centre is the last column of V
    x = vt[-1][0] / vt[-1][-1]
    y = vt[-1][1] / vt[-1][-1]
    z = vt[-1][2] / vt[-1][-1]
    c = np.array([x, y, z])  # camera centre

    # now we need to find M and K,R from M = KR.
    M = p[:3, :3]
    # assert np.allclose(-M@c, p[:,-1]), "-MC is not equal to last column of p for some reason..."

    # find K, R from RQ-decomposition of M
    K, R = sp.linalg.rq(M)

    # remove ambiguity in the RQ decomposition by making diagonals of K positive.
    T = np.diag(np.sign(np.diag(K)))
    assert np.allclose(K @ T @ T @ R, K @ R), "The transformation T is not its own inverse..."
    K = K @ T
    R = T @ R

    # ensure that the last diagonal of K is equal to 1
    scale = 1 / K[2, 2]
    K = scale * K

    return K, R, c

def decompose_essential_matrix(E):
    """Compute the decomposition of E = U diag(1,1,0) V^T.
    Return: U, np.diag(1,1,0), V^T"""
    u, d, vt = np.linalg.svd(E)

    # print(d)
    assert np.allclose(d[0], d[1]), "The first two elements of the diagonal are not equal"
    assert np.allclose(d[-1], 0), "The final diagonal element is not zero..."

    if not d[0] == 1:
        scale = 1 / d[0]  # force the diagonal elements to be (1,1,0)
        D = np.diag(scale * d)
        u = 1 / scale * u

    else:
        D = np.diag(d)
        pass

    assert np.allclose(u @ D @ vt, E), "SVD decomposition has not worked!"

    return u, D, vt

def compute_essential_matrix(K1, K2, F):
    """
    From the camera calibration matrices K1 and K2, and given the  \
    fundamental matrix F, compute and return the essential matrix E.
    """
    return K2.T @ F @ K1

def depth(P, X):
    """Given a 3D point X = (x, y, z) and P = [M|p_4] determine the depth
    in front of the camera.
    
    Return Depth of point X."""
    if not len(X) == 4:
        X = np.r_[X, 1]  # turn into homgeneous coordinate...

    T = X[-1]
    M = P[:3, :3]  # 3x3 left hand submatrix
    # p4 = P[:, -1]  # last column of p

    # now P(x, y, z, 1)^T = w(x,y,1)^T
    x = P @ X
    w = 1 / x[-1]

    depth = (np.sign(np.linalg.det(M)) * w) / (T * np.linalg.norm(M[:, 2]))

    return depth

def findCamerafromEssentialMTX(E, arbitrary3Dpoint):
    """Given E, compute the four possible camera matrices for camera 2. 
    Then we check which of these four solutions is physical with an arbitrarily chosen 
    computed 3D point. Return the two camera matrices..."""
    X = arbitrary3Dpoint

    u, _, vt = decompose_essential_matrix(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Z = np.array([[0,1,0], [-1,0,0], [0,0,0]])
    u3 = u[:, -1]

    P1 = np.c_[u @ W @ vt, u3]
    P2 = np.c_[u @ W @ vt, -u3]
    P3 = np.c_[u @ W.T @ vt, u3]
    P4 = np.c_[u @ W.T @ vt, -u3]

    cameras = [P1, P2, P3, P4]
    P0 = np.c_[np.eye(3), np.array([0, 0, 0])]
    assert depth(P0, X) > 0, "Original Camera cannot see the point!"

    for P in cameras:
        criteria = depth(P, X)
        if criteria > 0:
            return P0, P
        else:
            pass

    return None

def unit(vec):
    """Return the unit vector of vector."""
    return vec / np.linalg.norm(vec)

def linepoint2point(A, B, t):
    '''Find the equation of the line from point A -> B. \n
    Returns a point on the line P satisfies P = A + (B-A)*t for some t.'''
    return A + (B - A) * t

def findPlane(A, B, C, *args):
    '''Given three points A, B, C find the equation of the plane they all lie upon.
    Can input more points to verify if they are all coplanar. If testing the sensor 
    plane, this is recommended.
    \nReturns a,b,c of the normal=(a,b,c) and the scalar component np.dot(normal, A).'''
    normal = np.cross((B - A), (C - A))
    d = np.dot(normal, A)
    a, b, c = normal

    if args:
        for coord in args:
            assert np.allclose(a * coord[0] + b * coord[1] + c * coord[2] - d,
                               0), "additional input point is not coplanar..."

    return a, b, c, d

def point2plane2point(a, b, c, d, point3D, cameracentre):
    '''Where does the line joining the 3D point and camera centre intersect the plane? 
    Return the x,y,z position of this intercept. '''
    n = np.array([a, b, c])
    D = point3D
    E = cameracentre

    t = (d - np.linalg.multi_dot([n, D])) / np.linalg.multi_dot([n, (E - D)])

    # point of intersection is 
    x, y, z = D + (E - D) * t

    return x, y, z

def RotationMatrix(yaw, pitch, roll, rad=False):
    '''Form a rotation matrix to carry out specified yaw, pitch, roll rotation *in degrees*.'''
    Rx = R_x(pitch, rad=rad)
    Ry = R_y(yaw, rad=rad)
    Rz = R_z(roll, rad=rad)

    R = np.linalg.multi_dot([Rx, Ry, Rz])

    return R

def transformpoint(point, rot_matrix, tvec):
    '''Apply the transformations: rotation followed by translation to the point. '''
    newpoint = rot_matrix @ point
    newpoint += tvec
    return newpoint

def untransformpoint(transformedPoint, rot_matrix, tvec):
    '''Reverse the transformation of a point: translation followed by inverse of rotation...'''
    tpoint = transformedPoint - tvec
    point = np.linalg.inv(rot_matrix) @ tpoint

    return point

def pointInCamera1Image(point3D, sensor_width, sensor_height, focal_length, pixel_size,
                        cameracentre=np.array([0, 0, 0])):
    '''Can camera 1 (at the origin of the coordinate system) see the 3D point in space? \n
    Return True/False and the x,y pixel coordinates of the image.'''
    w, h = sensor_width, sensor_height
    f = focal_length

    TL = np.array([-w / 2, h / 2, f])  # top left of sensor
    TR = np.array([w / 2, h / 2, f])  # top right of sensor
    BR = np.array([w / 2, -h / 2, f])  # bottom right of sensor
    BL = np.array([-w / 2, -h / 2, f])  # bottom left of sensor

    # define limits on the sensor    (A point exceeding these dimensions cannot be seen)
    xmin, ymin, _ = BL
    xmax, ymax, _ = TR

    # define the plane of the sensor and the line linking the 3D point and the camera centre.
    # Where they intersect is the image coordinates of the image. -> then we can check whether it is in frame...

    # pick 3 corners of the four to define plane.
    a, b, c, d = findPlane(TL, TR, BR, BL)
    intersection = point2plane2point(a, b, c, d, point3D, cameracentre)

    # can the point be seen? 
    x, y, z = intersection

    # print(z, f)
    assert np.allclose(z, f), "Intersection of image plane hasn't worked properly... point not at z=f"

    seen = False
    if (x >= xmin and x <= xmax) and (y >= ymin and y <= ymax):
        seen = True

    # check if the point is in front of the CCD plane: 
    if point3D[2] < f:  # if the point's z-position is less than the z-position of the CCD plane
        seen = False  # then the point cannot be physically imaged by the camera.

    # return the pixel coordinates of the pixel... 
    # NOTE: we are treating the origin as the CENTRE of the image, not the bottom left corner. 
    x = x / pixel_size
    y = y / pixel_size

    return seen, x, y

def pointInCamera2Image(point3D, sensor_width, sensor_height, focal_length, pixel_size, cameracentre, tvec,
                        rotation_matrix):
    '''Can camera 2 at a given translation vector and rotation from camera 1 see the 3D point? \n
    Return True/False and x, y coordinate of point in the image.'''
    w, h = sensor_width, sensor_height
    f = focal_length

    # if the camera centre was at the origin and looking straight down the +ve z-direction
    TL = np.array([-w / 2, h / 2, f])  # top left of sensor
    TR = np.array([w / 2, h / 2, f])  # top right of sensor
    BR = np.array([w / 2, -h / 2, f])  # bottom right of sensor
    BL = np.array([-w / 2, -h / 2, f])  # bottom left of sensor
    sensor_centre = np.mean([TL, TR, BR, BL], axis=0)

    # define limits on the sensor
    xmin, ymin, _ = BL
    xmax, ymax, _ = TR

    # now apply transformation so everything is where it 'should be' in 3-space
    R = rotation_matrix
    newcameracentre = cameracentre  # don't need to transform this!
    TL2 = transformpoint(TL, R, tvec)
    TR2 = transformpoint(TR, R, tvec)
    BR2 = transformpoint(BR, R, tvec)
    BL2 = transformpoint(BL, R, tvec)
    new_sensor_centre = transformpoint(sensor_centre, R, tvec)

    # determine if point is in front or behind the camera
    point2camera = point3D - newcameracentre
    point2sensor = point3D - new_sensor_centre

    if np.linalg.norm(point2sensor) <= np.linalg.norm(point2camera):
        front = True
    else:
        front = False

    # define the plane of these points
    # also find intersection of the line from 3Dpoint and the camera centre, and the plane
    a, b, c, d = findPlane(TL2, TR2, BR2, BL2)
    intersection = point2plane2point(a, b, c, d, point3D, cameracentre)

    # this intersection is in the transformed frame. Untransform and we can see if it exceeds the 
    # sensor dimensions, and therefore whether the point can be seen on the CCD.

    x, y, z = untransformpoint(intersection, R, tvec)
    assert np.allclose(f, z), "Intersection of image plane hasn't worked properly... point not at z=f"

    seen = False
    if (x >= xmin and x <= xmax) and (y >= ymin and y <= ymax) and front:
        seen = True
    else:
        pass

    # turn the coordinates into pixel coordinates
    x = x / pixel_size
    y = y / pixel_size

    return seen, x, y

def fixImgCoords(imgx, imgy, sensor_width, sensor_height):
    '''Return the image coordinates as a np.array with the origin at the bottom left corner of the image.'''
    # TODO: This should quite possibly be the TOP LEFT corner! 
    # TODO: CHECK THIS CAREFULLY! 
    w, h = sensor_width, sensor_height
    origin = np.array([-w / 2, -h / 2])
    imgcoordinate = np.array([imgx, imgy])

    # relative to the origin, the point is at position:
    return imgcoordinate - origin
    # return np.array([imgx, imgy])

def stereoImages(x1s, y1s, x2s, y2s, w1, w2, h1, h2):
    '''Input: Arrays or lists of corresponding x1, y1, x2, y2 coordinates,
              on the two images of Camera 1 and Camera 2 respectively.
              CCD dimensions of Camera 1 (w1, h1) and Camera 2 (w2, h2)
              \n IN UNITS OF PIXELS...'''
    fig = plt.figure()
    ax1 = plt.subplot(121)
    plt.plot(x1s, y1s, "bx")
    # plot the sensor on
    plt.plot([0, w1], [0, 0], "r", alpha=0.3)  # bottom horizontal
    plt.plot([w1, w1], [0, h1], "r", alpha=0.3)  # right vertical
    plt.plot([w1, 0], [h1, h1], "r", alpha=0.3)  # top horizontal
    plt.plot([0, 0], [h1, 0], "r", alpha=0.3)  # left vertical

    ax1.set_aspect("equal")
    plt.xlabel("x-direction (pixels)")
    plt.ylabel("y-direction (pixels)")
    plt.title("Camera 1 Image")

    ax2 = plt.subplot(122, sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.plot(x2s, y2s, "bx")

    # plot the sensor on
    plt.plot([0, w2], [0, 0], "r", alpha=0.3)
    plt.plot([w2, w2], [0, h2], "r", alpha=0.3)
    plt.plot([w2, 0], [h2, h2], "r", alpha=0.3)
    plt.plot([0, 0], [h2, 0], "r", alpha=0.3)

    ax2.set_aspect("equal")
    plt.title("Camera 2 Image")
    plt.xlabel("x-direction (pixels)")

    plt.suptitle("Points as Imaged by Two Stereo-Cameras")
    fig.tight_layout()
    # plt.show()

def draw3D(points3D, cameracentre1, cameracentre2, TL1, TR1, BR1, BL1, TL2, TR2, BR2, BL2, tvec, rot_matrix,
           shrink_factor=1):
    # projected points for camera 1 
    prTL1 = findCoordinatesForZ(TL1, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)
    prTR1 = findCoordinatesForZ(TR1, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)
    prBR1 = findCoordinatesForZ(BR1, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)
    prBL1 = findCoordinatesForZ(BL1, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)

    # to find these for camera 2 we need to first assume camera is at origin also, then transform to new position...
    # untransform to origin...
    TL2 = untransformpoint(TL2, rot_matrix, tvec)
    TR2 = untransformpoint(TR2, rot_matrix, tvec)
    BR2 = untransformpoint(BR2, rot_matrix, tvec)
    BL2 = untransformpoint(BL2, rot_matrix, tvec)

    prTL2 = findCoordinatesForZ(TL2, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)
    prTR2 = findCoordinatesForZ(TR2, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)
    prBR2 = findCoordinatesForZ(BR2, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)
    prBL2 = findCoordinatesForZ(BL2, cameracentre1, findFurthestPoint(points3D, cameracentre1)[2] / shrink_factor)

    prTL2 = transformpoint(prTL2, rot_matrix, tvec)
    prTR2 = transformpoint(prTR2, rot_matrix, tvec)
    prBR2 = transformpoint(prBR2, rot_matrix, tvec)
    prBL2 = transformpoint(prBL2, rot_matrix, tvec)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = points3D[:, 0], points3D[:, 1], points3D[:, 2]

    ax.scatter(xs, ys, zs, c="b", label="3D points")
    ax.scatter(cameracentre1[0], cameracentre1[1], cameracentre1[1], c="g", label="Camera 1 Centre")
    ax.scatter(cameracentre2[0], cameracentre2[1], cameracentre2[2], c="m", label="Camera 2 Centre")
    # plot the 3D positions of the projected CCD corners
    ax.plot((prTL1[0], prTR1[0]), (prTL1[1], prTR1[1]), (prTL1[2], prTR1[2]), c="r", alpha=0.5, label="CCD Boundaries")
    ax.plot((prTR1[0], prBL1[0]), (prTL1[1], prBL1[1]), (prTR1[2], prBL1[2]), c="r", alpha=0.5)
    ax.plot((prBL1[0], prBR1[0]), (prBL1[1], prBR1[1]), (prBL1[2], prBR1[2]), c="r", alpha=0.5)
    ax.plot((prTL1[0], prBR1[0]), (prTL1[1], prBR1[1]), (prTL1[2], prBR1[2]), c="r", alpha=0.5)
    ax.plot((prTL1[0], cameracentre1[0]), (prTL1[1], cameracentre1[1]), (prTL1[2], cameracentre1[2]), c="y", alpha=0.5)
    ax.plot((prTR1[0], cameracentre1[0]), (prTR1[1], cameracentre1[1]), (prTR1[2], cameracentre1[2]), c="y", alpha=0.5)
    ax.plot((prBL1[0], cameracentre1[0]), (prBL1[1], cameracentre1[1]), (prBL1[2], cameracentre1[2]), c="y", alpha=0.5)
    ax.plot((prBR1[0], cameracentre1[0]), (prBR1[1], cameracentre1[1]), (prBR1[2], cameracentre1[2]), c="y", alpha=0.5)

    ax.plot((prTL2[0], prTR2[0]), (prTL2[1], prTR2[1]), (prTL2[2], prTR2[2]), c="r", alpha=0.5)
    ax.plot((prTR2[0], prBL2[0]), (prTL2[1], prBL2[1]), (prTR2[2], prBL2[2]), c="r", alpha=0.5)
    ax.plot((prBL2[0], prBR2[0]), (prBL2[1], prBR2[1]), (prBL2[2], prBR2[2]), c="r", alpha=0.5)
    ax.plot((prTL2[0], prBR2[0]), (prTL2[1], prBR2[1]), (prTL2[2], prBR2[2]), c="r", alpha=0.5)
    ax.plot((prTL2[0], cameracentre2[0]), (prTL2[1], cameracentre2[1]), (prTL2[2], cameracentre2[2]), c="y", alpha=0.5)
    ax.plot((prBL2[0], cameracentre2[0]), (prBL2[1], cameracentre2[1]), (prBL2[2], cameracentre2[2]), c="y", alpha=0.5)
    ax.plot((prTR2[0], cameracentre2[0]), (prTR2[1], cameracentre2[1]), (prTR2[2], cameracentre2[2]), c="y", alpha=0.5)
    ax.plot((prBR2[0], cameracentre2[0]), (prBR2[1], cameracentre2[1]), (prBR2[2], cameracentre2[2]), c="y", alpha=0.5)

    ax.legend()
    ax.axis("equal")
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")

def centroid(arr):
    """For a set of coordinates, compute the centroid position."""
    length = arr.shape[0]  # length of the array
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])

    return sum_x / length, sum_y / length

def translation(arr):
    """Produces a new set of points that has undergone a translation such that 
    the centroid of the original set of points now lies at the origin."""
    x_c, y_c = centroid(arr)
    newpoints = np.array((arr[:, 0] - x_c, arr[:, 1] - y_c))
    return newpoints.T

def avgDisplacement(arr):
    """Finds the average displacement of a set of points (x_i, y_i)
    from the origin."""
    sum_squares = np.sum(arr ** 2, axis=1)
    return np.mean(np.sqrt(sum_squares))

def scale(arr):
    """Fixes the set of points (x_i, y_i) such that the average 
    displacement from the origin is sqrt(2)."""
    displacement = avgDisplacement(arr)
    scale = np.sqrt(2) / displacement

    newpoints = scale * arr
    return newpoints

def normalisation(arr):
    """Given a set of points {x_i}, normalise them such that\n 
    a) the centroid of the group is at the origin \n
    b) the average displacement of the group is sqrt(2)\n
    return: The new set of normalised points corresponding to 
    original set of points."""

    # 1) translate the points such that the centroid is at origin
    points = translation(arr)
    # 2) scale the points such that average displacement is sqrt(2)
    newpoints = scale(points)

    return newpoints

def findCoordinatesForZ(A, B, z):
    '''Find the 3D coordinates of the point at a given z-value, for the point on the line 
    linking coordinates A and B.'''
    d = B - A
    l, m, n = d
    x1, y1, z1 = A

    y = m / n * (z - z1) + y1
    x = l / m * (z - z1) + x1

    return np.array([x, y, z])

def findFurthestPoint(arr_of_points, cameracentre):
    """Find the coordinate of the point that is furthest from the camera centre"""
    maxpoint = None
    maxdistance = 0
    for point in arr_of_points:
        distance = np.linalg.norm(point - cameracentre)
        if distance > maxdistance:
            maxdistance = distance
            maxpoint = point
        else:
            pass

    return maxpoint

def pickGroundTruthPoints(arr_of_points, no_ground_truths=5):
    """
    Randomly select the coordinates of 5 (default) 3D points to act \
    as ground truths to recover the metric reconstruction of the \
    projection...
    """
    index = []

    while len(index) < no_ground_truths:
        i = random.randint(0, len(arr_of_points) - 1)
        if not i in index:
            index.append(i)
            continue
        continue

    points = np.zeros((no_ground_truths, 3))

    # print(index)
    for i in range(len(index)):
        points[i] = arr_of_points[index[i]]

    return points, index
 
def getTransformationMatrices(coordinate1, coordinate2):
    """Define transformation matrices T1 and T2 that take coordinates
    (x1,y1) and (x2, y2) back to the origin, respectively."""
    x1, y1 = coordinate1[:2]
    x2, y2 = coordinate2[:2]

    x1Translation, y1Translation = centroid(np.array([[x1, y1]]))
    x2Translation, y2Translation = centroid(np.array([[x2, y2]]))

    T1 = np.array([[1, 0, -x1Translation], [0, 1, -y1Translation], [0, 0, 1]])
    T2 = np.array([[1, 0, -x2Translation], [0, 1, -y2Translation], [0, 0, 1]])

    # print(T2 @ coordinate2) # testcase, should yield (0,0,1) origin...

    return T1, T2

def translateFundamentalMatrix(F, T1, T2):
    """Replace matrix F with transpose(inv(T2)) F inv(T1)."""
    return np.linalg.inv(T2).T @ F @ np.linalg.inv(T1)

def findEpipoles(F):
    """
    Find the right and left epipoles e1 and e2 such that 
    e'.T F = 0 and F e = 0. 
    
    Return normalsed epipoles...
    """

    # e1 is the right null-space of F, e2.T is left null space, so use SVD. 
    e1 = right_null_space(F)
    e2 = left_null_space(F).T
    E2 = right_null_space(F.T)

    e1 = sp.linalg.null_space(F)[:,0]
    e2 = sp.linalg.null_space(F.T)[:,0]

    # print(F)

    # print(np.allclose(F.T @ E2, 0), np.allclose(e2.T @ F, 0))

    # print(e1, e2)
    

    assert np.isclose(np.linalg.norm(np.matmul(e2.T, F)), 0), "First epipole DOESN'T satisfy the epipolar constraint"
    assert np.isclose(np.linalg.norm(np.matmul(F, e1)), 0), "Second epipole DOESN'T satisfy the epipolar constraint" 

    # print(F @ e1)
    # print("e1 is ", e1)
    # print("e2 is ", e2)
    # print("E2 is ", E2, "\n")

    # normalise the epipoles such that (e11**2 + e12**2 = 1, and e21**2 + e22**2 = 1)
    scaleFactor1 = 1/(e1[0]**2 + e1[1]**2)**.5
    scaleFactor2 = 1/(e2[0]**2 + e2[1]**2)**.5
    scaleFactor3 = 1/(E2[0]**2 + E2[1]**2)**.5

    newE2 = scaleFactor3*E2

    # print("The scale factors for the epipoles are: {} and {}".format(scaleFactor1, scaleFactor2))

    newe1, newe2 = scaleFactor1*e1, scaleFactor2*e2

    return newe1, newE2

def R_x(angle, rad=False):
    if not rad:   # convert to radians if in degrees
        angle = np.deg2rad(angle)
    return np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])

def R_y(angle, rad=False):
    if not rad:
        angle = np.deg2rad(angle)
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

def R_z(angle, rad=False):
    if not rad:
        angle = np.deg2rad(angle)
    return np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

def getRotationMatrices(e1, e2):
    """Given two epipoles e1 = (e11, e12, e13)^T and 
    e2 = (e21, e22, e23)^T, return rotation matrices R
    and R' such that Re1 = (1, 0, e13)^t and Re2 = (1,0,e23)^T)."""
    e11, e12, _ = e1.ravel()
    e21, e22, _ = e2.ravel()

    R1 = np.array([[e11, e12, 0], [-e12, e11, 0], [0, 0, 1]])
    R2 = np.array([[e21, e22, 0], [-e22, e21, 0], [0, 0, 1]])

    # print(np.isclose((R1 @ e1)[0], 1))

    assert np.allclose(R2 @ e2, np.array([1, 0, e2[2]])), "R2 doesn't give the required R @ e2 = (1, 0, e2_3)"
    assert np.allclose(R1 @ e1, np.array([1, 0, e1[2]])), "R1 doesn't give the required R @ e1 = (1, 0, e1_3)"

    return R1, R2

def rotateFundamentalMatrix(F, R1, R2):
    """Replace matrix F with (R2 F R1.T)"""
    return R2 @ F @ R1.T

def test_form_of_F(F, newe1, newe2):
    a, b, c, d = F[1,1], F[1,2], F[2,1], F[2,2]
    f, g = newe1[2], newe2[2]

    # print("F is: \n", F)
    # theory = np.array([[f*g*d, -g*c, -g*d], [-f*b, a, b], [-f*d, c, d]])

    # print("F should be: \n", theory)

    wrong = 0
    if not np.isclose(f*g*d, F[0,0]):
        print("F[0,0] isn't correct form")
        wrong += 1
    if not np.isclose(-g*c, F[0,1]):
        print("F[0,1] isn't correct form")
        wrong += 1
    if not np.isclose(-g*d, F[0,2]):
        print("F[0,2] isn't correct form")
        wrong += 1
    if not np.isclose(-f*b, F[1,0]):
        print("F[1,0] isn't correct form")
        wrong += 1
    if not np.isclose(-f*d, F[2,0]):
        print("F[2,0] isn't correct form")
        print("F[2,0] is {}, should be {}".format(F[2,0], -f*d))
        wrong += 1

    # print("Failed on {}/5 elements of matrix F".format(wrong))
    return None

def g(t, f1, f2, a, b, c, d):
    term1 = t*((a*t+b)**2 + f2**2*(c*t + d)**2)**2
    term2 = (a*d - b*c)*(1 + f1**2 * t**2)**2 * (a*t+b)*(c*t+d)

    return term1 - term2

def formPolynomial(e1, e2, F):
    """Given two epipoles e1=(e11,e12,e13)^T and 
    e2=(e21, e22, e23)^T and fundamental matrix F, 
    form polynomial g(t)... """
    # print("e1 is ", e1)
    # print("e2 is ", e2)
    # print("F = ", F)
    # extract parameters from epipoles and matrix F
    f = e1[2]    # f
    g = e2[2]    # f' in literature
    _, _, _, _, F_22, F_23, _, F_32, F_33 = F.ravel()  
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
        val = np.polyval(coefs, t)
        if np.isclose(val, 0):
            counter += 1
    assert counter > 0, "No roots were found..."

    return roots

def costFunction(t, a, b, c, d, f, g):

    return (t**2/(1+f**2*t**2)) + (c*t+d)**2/((a*t+b)**2 + g**2*(c*t+d)**2) 

def evaluateCostFunction(roots, a, b, c, d, f, g):
    """Evaluate the cost function (12.5) for input array/list np.array([t1,t2, ...]).\n
    Select the value of t for which the cost function is the smallest."""

    tmin = 9e15
    minCostFn = 9e15
    
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

def findModelPoints(tmin, a, b, c, d, f, g ):
    """Find the model points x and x' that fit the epipolar constrant x'^T F x = 0\n
    Return the coordinates x1,x2"""

    l1 = np.array([tmin*f, 1, -tmin])
    l2 = np.array([-g*(c*tmin+d), a*tmin+b, c*tmin+d])

    x1 = np.array([-l1[0]*l1[2], -l1[1]*l1[2], l1[0]**2 + l1[1]**2])
    x2 = np.array([-l2[0]*l2[2], -l2[1]*l2[2], l2[0]**2 + l2[1]**2])

    return x1, x2

def findOriginalCoordinates(R1, R2, T1, T2, x1, x2):
    """Transfer the normalised coordinates back to the original coordinates using the formulas:\n
    x1 = T1^-1 R1^T x1; x2 = T2^-1 R2^T x2"""

    newx1 = np.linalg.inv(T1) @ R1.T @ x1
    newx2 = np.linalg.inv(T2) @ R2.T @ x2

    return newx1, newx2

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

    _, d, vt = np.linalg.svd(A)
    i = np.argmin(d)
    X = vt[i]
    # or can use right_null_space(A).reshape(4,1)

    return X

def homogeneous2Inhomogeneous(X):
    """Convert homogeneous coordinate [x1, x2, x3, ..., xn] to \ninhomogeneous coordinate
    [x1/xn, x2/xn, ..., xn-1/xn]"""
    divider = X[-1]
    # print(X[len(X)-2])

    newcoord = np.zeros(len(X)-1)

    for i in range(len(X)-1):
        newcoord[i] = (X[i]/divider)

    return newcoord

def triangulate(imgcoord1, imgcoord2, P1, P2):
    '''Triangulate the coordinates in each image back to a 3D point X, using their 
    projection matrices P1 and P2 respectively...
    
    Return an INHOMOGENEOUS 3-vector coordinate of the point \
    corresponding to the image in camera 1 with camera matrix P1 and \
    the image in camera 2 with camera matrix P2. '''
    # first form the matrix A
    A = formA(imgcoord1, imgcoord2, P1, P2)
    
    # find point X in homogeneous coordinates
    X = find3DPoint(A)

    # print(A@X)

    # print("Does AX=0? ", np.allclose(A@X, 0))

    # modify back to inhomogeneous (3-vector) coordinate

    return homogeneous2Inhomogeneous(X)

def getTransformMtx(arr_of_points):
    '''Find the transformation that normalises a group of homogeneous coordinates'''
    x_c, y_c = centroid(arr_of_points)

    sumSquares = np.sum((arr_of_points[:,0]-x_c)**2 + (arr_of_points[:,1]-y_c)**2)

    s = np.sqrt(sumSquares/(2*len(arr_of_points)))

    T = np.array([[1/s, 0, -1/s * x_c], [0, 1/s, -1/s * y_c], [0,0,1]])

    # print(T)

    return T 

def formMatrixA(arr_of_imgpoints1, arr_of_imgpoints2):
    '''Form the matrix A as in (11.3, p279).'''

    assert len(arr_of_imgpoints1) == len(arr_of_imgpoints2), "The array of image points must be matching and the same length"

    if len(arr_of_imgpoints1[0]) == 3:
        true = 0
        for i in range(len(arr_of_imgpoints1)):
            if np.isclose(arr_of_imgpoints1[i][-1], 1) and np.isclose(arr_of_imgpoints2[i][-1], 1):
                true += 1
        
        assert true == len(arr_of_imgpoints1), "Homogeneous coordinates not ending in 1!"

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
        col8.append(y1)
    
    A[:,0], A[:,1], A[:,2], A[:,3], A[:,4],A[:,5], A[:,6], A[:,7]  = col1, col2, col3, col4, col5, col6, col7, col8

    return A

def solveFundamentalMatrix(A):
    '''For set of linear equations in matrix A, find f to solve Af=0. \n
    Return F, the matrix form of 9-vector f. '''
    # Find right null space of A (use SVD)
    F = right_null_space(A)
    
    # Now we need to constrain that det(F)=0, need unique solution! 
    # Take SVD of F
    u, d, v = np.linalg.svd(F)

    # Check d = diag(r,s,t), with r >= s >= t. 
    r, s, t = d
    assert r >= s, "The SVD of F has produced a D = diag(r,s,t) where the contraints r >= s >= t have NOT been met..."
    assert s >= t, "The SVD of F has produced a D = diag(r,s,t) where the contraints r >= s >= t have NOT been met..."

    # if this criteria is met then the minimised F = U diag(r,s,0) V^T
    D = np.diag([r, s, 0])

    newF = u @ D @ v

    d= np.linalg.svd(newF, compute_uv=False)
    # print("Diagonal elements of newF are : ", d)

    # print("F = ", newF)

    # print("F has rank({})".format(np.linalg.matrix_rank(newF)))

    assert np.linalg.matrix_rank(newF) == 2, "Computed F has rank({}), the correct F should have rank(2)...".format(np.linalg.matrix_rank(newF))
    assert np.isclose(np.linalg.det(newF), 0), "Computed F is not singular, i.e det(F) != 0..."
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
    e2 = right_null_space(F.T) 


    print("epipole e2 = ", e2)
    print("Epipole found correctly? ", np.allclose(F.T @ e2, 0))

    e2_skew = skew(e2)
    print("Is e2 it's skew matrix's null vector? ", np.allclose(e2_skew @ e2, 0))

    # print("The skew matrix e2 is ", e2_skew)

    leftMatrix = e2_skew @ F 

    print("left submatrix skew(e2)F is rank({})".format(np.linalg.matrix_rank(leftMatrix)))
    P2 = np.c_[skew(e2)@F, e2]

    # print("skew epipole = ", e2_skew)
    # print("skew(e2) @ F = ", leftMatrix)
    # print("Epipole = ", e2)
    # print("P2 = ", P2)

    return P1, P2

def cameraMatrices(img1_points, img2_points):
    """
    Determine the camera matrices p1, p2 and the fundemental matrix \
    relating image correspondences in image 1 and image 2. \
    This carries out the normalised 8-point algorithm as in Hartley & \
    Zisserman .

    Inputs:
    img1_points : np.array of image coordinates in first camera.
    img2_points : np.array of image coordinate in second camera.
    NOTE : The points must be corresponding, and in the same order, in 
           img1_points and img2_points. 

    Out: P1, P2, F
    """
    img1coords = img1_points
    img2coords = img2_points

    # define the transformation matrices that normalise the coordinates...
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

    # # TESTCASE, remove if heavy...
    # true = 0
    # for i in range(len(img1coords)):
    #     if np.isclose(img2coords[i] @ F @ img1coords[i], 0):
    #         true += 1
    #     else:
    #         pass

    # assert true == len(img1coords), "The equation x'Fx=0 did not hold for all x', x"

    p1, p2 = findCameras(F)

    return p1, p2, F

def estimate3DPoints(img1_points, img2_points):
    """
    Use the optimal triangulation algorithm (12.1, p. 318), to 
    """
    
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

        print(newF)

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

def optimal_triangulation(img1coords, img2coords, F):
    """
    Use the optimal triangulation algorithm (12.1, p. 318), to compute
    the optimal 3D points corresponding to the matched image coordinates
    of camera 1 and camera 2 and fundamental matrix F. 

    Parameters: 
    img1coords : numpy array of homogeneous 3-vectors of the image \
                 coordinates in image 1.
    img2coords : numpy array of homogeneous 3-vectors of the image \
                 coordinate in image 2. 
    F : 3 x 3 numpy matrix, the optimal fundamental matrix pre-computed \
    from n >= 8 point correspondences in image 1 and 2.  

    Return: np.array of inhomogeneous 3D points corresponding to the \
            img1 and img2 coordinates.
    """
    # Make array to store the reconstructed 3D positions...
    triangulated_points = np.zeros((len(img1coords),3))
    p1, p2 = findCameras(F) # compute set of cameras corresponding to F

    # Compute the optimal point correspondences that minimise geometric error
    for i in range(len(img1coords)):
        T1, T2 = getTransformationMatrices(img1coords[i], img2coords[i]) # transformation that takes back x1, x2 back to origin
        newF = translateFundamentalMatrix(F, T1, T2)
        e1, e2 = findEpipoles(newF)
        R1, R2 = getRotationMatrices(e1, e2)
        newF = rotateFundamentalMatrix(newF, R1, R2)
        test_form_of_F(newF, e1, e2)   # test the form of F
        a,b,c,d,f,g = formPolynomial(e1, e2, newF)
        roots = solvePolynomial(a,b,c,d,f,g)
        tmin = evaluateCostFunction(roots, a,b,c,d,f,g)
        x1, x2 = findModelPoints(tmin,a,b,c,d,f,g)  # These are the corrected point correspondences
        newx1, newx2 = findOriginalCoordinates(R1, R2, T1, T2, x1, x2) # transform back to original coordinates.
        
        # check that x'Fx = 0 still true for new optimal x1, x2
        assert np.allclose(newx2.T @ F @ newx1, 0), "epipolar constraint failed with new optimal image coordinates"

        # print(np.allclose(homogeneous2Inhomogeneous(img2coords[i]), homogeneous2Inhomogeneous(newx2)))
        # newx1 and newx2 are the optimal point correspondences! 
        # Now use homogeneous triangulation method to compute 3D point.
        X = triangulate(newx1, newx2, p1, p2)

        # print(homogeneous2Inhomogeneous(newx2), homogeneous2Inhomogeneous(p2 @ np.append(X, [1])))
        # print("newx1 = P1 X ? ", np.allclose(homogeneous2Inhomogeneous(newx1), homogeneous2Inhomogeneous(p1 @ np.append(X, [1]))))
        # print("newx2 = P2 X ? ", np.allclose(homogeneous2Inhomogeneous(newx2), homogeneous2Inhomogeneous(p2 @ np.append(X, [1]))))

        # print("{} should roughly equal {}".format(homogeneous2Inhomogeneous(newx2), homogeneous2Inhomogeneous(p2 @ np.append(X, [1]))))
        triangulated_points[i] = X
    
    return triangulated_points 

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

        while A.shape[0] < 16:
            A = np.r_[A, np.zeros((1,16))]
            # print(A.shape)


        return A

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
    # # First check whether the points X_i and X_Ei are homogeneous or not...
    # if not len(X_i[0]) == 4:
    #     X_i = np.c_[X_i, np.ones(len(X_i))]
    # if not len(X_Ei[0]) == 4:
    #     X_Ei = np.c_[X_Ei, np.ones(len(X_Ei))]
    
    # print(X_i, X_Ei)

    # print(len(X_Ei), len(X_i))

    H = DLT(X_i, X_Ei)

    # print("Is X_Ei = H X_i? ")
    true = 0
    for i in range(len(X_Ei)):
        X_measured = np.append(X_i[i], [1])
        X_actual = np.append(X_Ei[i], [1])
        # print(X_measured, X_actual)
        # print(homogeneous2Inhomogeneous(H @ X_measured), homogeneous2Inhomogeneous(X_actual))
        if np.allclose(homogeneous2Inhomogeneous(H @ X_measured), homogeneous2Inhomogeneous(X_actual)):
            true += 1
        else:
            pass
    
    # print("True for {}/{} points".format(true, len(X_Ei)))

    # print("H = ", H)
    P1 = P1 @ np.linalg.inv(H)
    P2 = P2 @ np.linalg.inv(H)

    if not len(all_projected_points[1]) == 4:
        all_projected_points = np.c_[all_projected_points, np.ones(len(all_projected_points))]
    newpoints = np.zeros(all_projected_points.shape)

    for i in range(len(newpoints)):
        newpoints[i] = H @ all_projected_points[i]

    
    return P1, P2, newpoints

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
    H = right_null_space(A)

    Hnew = np.linalg.inv(T2) @ H @ T1

    # print(A @ Hnew.reshape(16,1))

    return Hnew

def convert_to_array(x1s, y1s, x2s, y2s):
    """
    Convert list of image match points to a np.array of homogeneous coordinates. 

    Return: img1coords, img2coords (Homogeneous coordinates)
    """
    assert len(x1s) == len(x2s), "The image correspondences must have the same length!"

    img1coords = np.zeros((len(x1s), 3))
    img2coords = np.zeros(img1coords.shape)
    img1coords[:,2], img2coords[:,2] = 1,1

    for i in range(len(x1s)):
        img1coords[i][0] = x1s[i]
        img1coords[i][1] = y1s[i]
        img2coords[i][0] = x2s[i]
        img2coords[i][1] = y2s[i]

    return img1coords, img2coords

def right_null_space(A):
    """
    Compute the right null space of Ah = 0, i.e find vector h, using SVD
    of A, then the least-squares solution subject to the normalisation
    that ||h|| = 1. 

    Input: 
    A : m x n matrix 

    Return n x n square matrix H.  
    """

    m, n = A.shape
    
    # Ensure that A has at least as many rows as columns...
    while n > m:
        A = np.r_[A, np.zeros((1,n))]
    
    # Compute the SVD of A
    _, d, vt = np.linalg.svd(A)
    i = np.argmin(d)
    h = vt[i] 

    # Form a square matrix H
    if len(h) < 4:
        return h
    H = h.reshape(int(np.sqrt(len(h))), int(np.sqrt(len(h))))

    return H

def left_null_space(A):
    """
    Compute the left null space of matrix A, corresponding to vector h \
    satisfying hA=0 under the condition that ||h||=1. 

    Return 
    """
    m, n = A.shape
    
    # Ensure that A has at least as many rows as columns...
    while n > m:
        A = np.r_[A, np.zeros((1,n))]

    u, d, _ = np.linalg.svd(A)
    i = np.argmin(d)
    f = u[:,i]   # i-th column of u 

    return f

def r(x):
    """Return the 2D distance to the origin of point x."""
    return np.sqrt(np.sum(x**2))

def transform_2d(arr_coordinates_2d):
    """
    determine the transformation matrix T for which Tx=x_hat and \
    the centroid of the set of coordinates {x_hat} is the origin \
    and the average distance from the origin of x_hat is sqrt(2). 

    Return T
    """
    xs = arr_coordinates_2d
    # print(xs)
    assert len(xs[2]) == 2, "Must be inhomogeneous 2-vectors"

    # Determine the average x and y point.
    x_c, y_c = np.mean(xs, axis=0)

    # Now we want to scale the points such that the mean distance from 
    # the origin is sqrt(2).
    T1 = np.array([[1, -x_c/y_c], [-y_c/x_c, 1]])

    tf_xs = np.zeros(xs.shape)
    for i in range(len(xs)):
        tf_xs[i] = T1 @ xs[i]
    
    mean = 0.
    for x in xs:
        mean += r(T1 @ x)
    mean = mean/len(xs)

    sf = mean/np.sqrt(2)

    # overall translation is T
    T = np.array([[1/sf, -x_c/(sf*y_c)], [-y_c/(sf*x_c), 1/sf]])

    return T

def _perturb_camera_matrix(K, R, centre, f_error, x_error, y_error):
    """
    For a given camera matrix P = KR[I|-C], perturb the camera \
    callibration matrix K elements randomly by amount equal to random \
    interval [-1,1]*element_uncertainty. 

    FOR USE IN MONTE CARLO SIMULATION...

    Return: Perturbed P matrix...
    """
    f = K[0,0]
    x = K[0,2]
    y = K[1,2]

    new_f = f + random.uniform(-1,1)*f_error
    new_x = x + random.uniform(-1,1)*x_error
    new_y = y + random.uniform(-1,1)*y_error

    new_K = np.array([[new_f, 0, x], [0, new_f, y], [0,0,1]])

    new_P = new_K @ R @ np.c_[np.eye(3), -centre.reshape(3,1)]

    return new_P

def uniform_sphere(n_points, radius=1):
    """
    Return a uniform distribution of points on the surface of a sphere.

    Parameters:
    n_points = number of points on the sphere
    radius = Radius of the sphere, default is 1 unit.

    Return xs, ys, zs
    """
    R = radius
    thetas = np.zeros(n_points)
    phis   = np.zeros(n_points)

    for i in range(n_points):
        thetas[i] = random.uniform(0,1) * 2*np.pi 
        phis[i]   = np.arccos(1-2*random.uniform(0,1))

    xs = R * np.sin(phis)*np.cos(thetas)
    ys = R * np.sin(phis)*np.sin(thetas)
    zs = R * np.cos(phis)

    return xs, ys, zs

def antipodes(xs, ys, zs):
    """
    Find the x,y,z coordinates opposite to input x, y, z coordinates, \
    the antipode of the sphere
    """
    return -xs, -ys, -zs

def position_sphere(sphere_centre, n_points, radius=1, origin=np.array([0,0,0])):
    """
    Place a sphere with n-uniformly distributed points at position \
    specified. 

    Parameters:: 
    sphere_centre : 3D coordinate of the centre of the sphere.
    n_points : number of uniform points to generate on the sphere.
    radius : radius of the sphere, default is 1 unit. 
    origin : origin of the coordinate system, default is (0,0,0).

    Return np.array([[x_1,y_1,z_1], ..., [x_n, y_n, z_n]]) coordinates \
    of points on the sphere and their antipodes: -points coordinates. 
    """
    xs, ys, zs = uniform_sphere(n_points, radius=radius)
    tvec = sphere_centre-origin
    points = np.zeros((n_points, 3))
    for i in range(n_points):
        points[i][:] = xs[i], ys[i], zs[i]

    antipoints = -points
    points += tvec
    antipoints += tvec
    

    return points, antipoints

def functional_error(f, f_err, px, px_err, py, py_err, camera2_x, 
                    camera2_x_err, camera2_y, camera2_y_err, camera2_z, 
                    camera2_z_err, yaw, yaw_err, pitch, pitch_err,
                    roll, roll_err, u1, u2, u_err, v1, v2, v_err):
    """
    Perform a functional approach on the camera 1 image coordinate u,v \
    corresponding with image point u',v' in the second cameara.

    PARAMETERS
    f : focal length in PIXELS
    f_err : uncertainty on the focal length in PIXELS
    px : principal point x-coordinate in PIXELS
    px_err : uncertainty on px in PIXELS
    py : principal point y-coordinat in PIXELS
    py_err : uncertainty on py in PIXELS
    camera2_x, camera2_y, camera2_z : Camera 2 position relative to \
                                      camera 1 (length units)
    camera2_x_err etc : uncertainty on camera 2 position (length units)
    yaw : yaw angle in DEGREES
    yaw_err : uncertainty on yaw angle in DEGREES
    pitch : pitch angle in DEGREES
    pitch_err : uncertainty on the pitch angle in DEGREES
    roll : roll angle in DEGREES
    roll_err : uncertainty on roll angle in DEGREES
    u1 : x-coordinate of image in camera 1
    u2 : x-coordinate of image in camera 2
    v1 : y-coordinate of image in camera 1
    v2 : y-coordinate of image in camera 2

    Return mean_value, absolute error in the reconstructed point.
    """
    K = np.array([[f, 0, px], [0, f, py], [0,0,1]])
    R2 = RotationMatrix(yaw, pitch, roll)
    R1 = np.eye(3)
    centre2 = np.array([camera2_x, camera2_y, camera2_z])
    centre1 = np.zeros(shape=(3,))

    p1 = K @ R1 @ np.c_[np.eye(3), -np.zeros((3,1))]
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    imgpoint1, imgpoint2 = [u1, v1], [u2, v2]
    
    mean_X = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2))

    p1f = np.array([[f+f_err, 0, px], [0,f+f_err, py], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2f = np.array([[f+f_err, 0, px], [0,f+f_err, py], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1x1 = np.array([[f, 0, px+px_err], [0,f, py], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2x2 = np.array([[f, 0, px+px_err], [0,f, py], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1y1 = np.array([[f, 0, px], [0,f, py+py_err], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2y2 = np.array([[f, 0, px], [0,f, py+py_err], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2cx = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0] + camera2_x_err], [centre2[1]], [centre2[2]]])]
    p2cy = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0]], [centre2[1]+camera2_y_err], [centre2[2]]])]
    p2cz = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0]], [centre2[1]], [centre2[2]+camera2_z_err]])]
    p2_yaw = K @ RotationMatrix(yaw+yaw_err, pitch, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_pitch = K @ RotationMatrix(yaw, pitch+pitch_err, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_roll = K @ RotationMatrix(yaw, pitch, roll+roll_err) @ np.c_[np.eye(3), -centre2.reshape(3,1)]


    alpha_f1 = np.asarray(triangulate(imgpoint1, imgpoint2, p1f, p2) ) - mean_X
    alpha_y1 = np.asarray(triangulate(imgpoint1, imgpoint2, p1y1, p2) ) - mean_X
    alpha_f2 = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2f) ) - mean_X
    alpha_x1 = np.asarray(triangulate(imgpoint1, imgpoint2, p1x1, p2) ) - mean_X
    alpha_x2 = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2x2) ) - mean_X
    alpha_y2 = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2y2) ) - mean_X
    alpha_xc = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2cx)  ) - mean_X
    alpha_yc = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2cy) ) - mean_X
    alpha_zc = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2cz) ) - mean_X
    alpha_alpha = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2_yaw)) - mean_X
    alpha_beta = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2_pitch) ) - mean_X
    alpha_gamma= np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2_roll) ) - mean_X
    alpha_img1x = np.asarray(triangulate([imgpoint1[0]+u_err, imgpoint1[1]], imgpoint2, p1, p2) ) - mean_X
    alpha_img1y = np.asarray(triangulate([imgpoint1[0], imgpoint1[1]+v_err], imgpoint2, p1, p2) ) - mean_X
    alpha_img2x = np.asarray(triangulate(imgpoint1, [imgpoint2[0]+u_err, imgpoint2[1]], p1, p2) ) - mean_X
    alpha_img2y = np.asarray(triangulate(imgpoint1, [imgpoint2[0], imgpoint2[1]+v_err], p1, p2) ) - mean_X

    total_err = np.sqrt(alpha_f1**2 + alpha_f2**2 + alpha_x1**2 + alpha_x2**2 + alpha_y1**2 + alpha_y2**2 +  
                alpha_xc**2 + alpha_yc**2 + alpha_zc**2 + alpha_alpha**2 + alpha_beta**2 + alpha_gamma**2 +
                alpha_img1x**2 + alpha_img1y**2 + alpha_img2x**2 + alpha_img2y**2)
    # print(mean_X, " +- ", total_err)

    return mean_X, total_err

def nearest_neighbour_err(f, f_err, px, px_err, py, py_err, camera2_x, 
                    camera2_x_err, camera2_y, camera2_y_err, camera2_z, 
                    camera2_z_err, yaw, yaw_err, pitch, pitch_err,
                    roll, roll_err, u_err, v_err, 
                    sphere_centre, n_points, radius=1, origin=np.array([0,0,0])):
    """
    Find the average error in nearest neighbour distance over all orientations \
    of two birds
    """
    # Generate uniform sphere surrounding specified position with antipodes
    points, antipoints = position_sphere(sphere_centre, n_points, 
                                         radius=radius, origin=origin)

    # Define camera matrices P1 and P2 
    K = np.array([[f, 0, px], [0, f, py], [0,0,1]])    
    R2 = RotationMatrix(yaw, pitch, roll)
    R1 = np.eye(3)
    centre2 = np.array([camera2_x, camera2_y, camera2_z])

    p1 = K @ R1 @ np.c_[np.eye(3), -np.zeros((3,1))]
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]

    rc_points = np.zeros(points.shape)
    rc_points_err = np.zeros(points.shape)
    rc_antipoints = np.zeros(antipoints.shape)    
    rc_antipoints_err = np.zeros(antipoints.shape)
    for i in range(len(points)):
        # Project the point onto the camera images
        u1, v1 = homogeneous2Inhomogeneous(p1 @ np.append(points[i], [1]))
        u2, v2 = homogeneous2Inhomogeneous(p2 @ np.append(points[i], [1]))

        # Reconstruct the point, get error using functional approach
        rc_point, rc_point_err = functional_error(f, f_err, px, px_err, py, py_err, camera2_x, 
                                        camera2_x_err, camera2_y, camera2_y_err, camera2_z, 
                                        camera2_z_err, yaw, yaw_err, pitch, pitch_err,
                                        roll, roll_err, u1, u2, u_err, v1, v2, v_err)
        rc_points[i] = rc_point
        rc_points_err[i] = rc_point_err
        # Project the point onto the camera images
        u1, v1 = homogeneous2Inhomogeneous(p1 @ np.append(antipoints[i], [1]))
        u2, v2 = homogeneous2Inhomogeneous(p2 @ np.append(antipoints[i], [1]))
        rc_antipoint, rc_antipoint_err = functional_error(f, f_err, px, px_err, py, py_err, camera2_x, 
                                        camera2_x_err, camera2_y, camera2_y_err, camera2_z, 
                                        camera2_z_err, yaw, yaw_err, pitch, pitch_err,
                                        roll, roll_err, u1, u2, u_err, v1, v2, v_err)
        # print(rc_point, rc_antipoint)
        rc_antipoints[i] = rc_antipoint
        rc_antipoints_err[i] = rc_antipoint_err
    
    # Nearest neighbour distance
    nnd = np.zeros(len(rc_antipoints))
    for i in range(len(rc_points)):
        nnd[i] = np.linalg.norm((rc_points[i]+rc_points_err[i])-(rc_antipoints[i]+rc_antipoints_err))

    # print("mean error in reconstructed points on sphere :", np.mean(rc_points_err, axis=0))
    # print("mean error in reconstructed antipodes on sphere :", np.mean(rc_antipoints_err, axis=0))

    # mean errors from points and antipodes on the sphere's surface [x,y,z]
    errors = np.mean(np.mean([rc_points_err, rc_antipoints_err], axis=1), axis=0)
    

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # ax.scatter(points[:,0], points[:,1], points[:,2], c="r", s=3)
    # ax.scatter(antipoints[:,0], antipoints[:,1], antipoints[:,2], c="b", s=3)

    # for i in range(len(points)):
    #     ax.plot((points[i][0], antipoints[i][0]), (points[i][1], antipoints[i][1]), (points[i][2], antipoints[i][2]), c="g")

    # plt.show()

    return errors

def relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2):
    return np.sum(np.sqrt((triangulate(imgpoint1_cam1, imgpoint1_cam2, p1, p2) - 
                  triangulate(imgpoint2_cam1, imgpoint2_cam2, p1, p2))**2))

def nnd_err(f, f_err, px, px_err, py, py_err, camera2_x, 
                    camera2_x_err, camera2_y, camera2_y_err, camera2_z, 
                    camera2_z_err, yaw, yaw_err, pitch, pitch_err,
                    roll, roll_err, u_err, v_err, 
                    sphere_centre, n_points, radius=1, origin=np.array([0,0,0])):
    """
    Find the average error in nearest neighbour distance over all orientations \
    of two birds
    """
    # Generate uniform sphere surrounding specified position with antipodes
    points, antipoints = position_sphere(sphere_centre, n_points, 
                                         radius=radius, origin=origin)

    K = np.array([[f, 0, px], [0, f, py], [0,0,1]])  # calibration matrix, same for both cameras
    R2 = RotationMatrix(yaw, pitch, roll)            # rotation matrix for camera 2
    R1 = np.eye(3)                                   # rotation matrix for camera 1
    centre2 = np.array([camera2_x, camera2_y, camera2_z])   # centre of camera 2
    centre1 = np.zeros((3,1))
    p1 = K @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]        # camera matrix 1
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]   # camera matrix 2

    # define the perturbed camera matrices for functional approach
    p1f = np.array([[f+f_err, 0, px], [0,f+f_err, py], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2f = np.array([[f+f_err, 0, px], [0,f+f_err, py], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1x1 = np.array([[f, 0, px+px_err], [0,f, py], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2x2 = np.array([[f, 0, px+px_err], [0,f, py], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1y1 = np.array([[f, 0, px], [0,f, py+py_err], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2y2 = np.array([[f, 0, px], [0,f, py+py_err], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2cx = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0] + camera2_x_err], [centre2[1]], [centre2[2]]])]
    p2cy = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0]], [centre2[1]+camera2_y_err], [centre2[2]]])]
    p2cz = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0]], [centre2[1]], [centre2[2]+camera2_z_err]])]
    p2_yaw = K @ RotationMatrix(yaw+yaw_err, pitch, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_pitch = K @ RotationMatrix(yaw, pitch+pitch_err, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_roll = K @ RotationMatrix(yaw, pitch, roll+roll_err) @ np.c_[np.eye(3), -centre2.reshape(3,1)]

    # now, for every point and antipode, compute the relative distance error
    # then average over all points

    total_errors = np.zeros(len(points))
    f1_errors = np.zeros(len(points))
    f2_errors = np.zeros(len(points))
    x0_1_errors = np.zeros(len(points))
    x0_2_errors = np.zeros(len(points))
    y0_1_errors = np.zeros(len(points))
    y0_2_errors = np.zeros(len(points))
    yaw_errors = np.zeros(len(points))
    pitch_errors = np.zeros(len(points))
    roll_errors = np.zeros(len(points))
    xc_errors = np.zeros(len(points))
    yc_errors = np.zeros(len(points))
    zc_errors = np.zeros(len(points))
    u11_errors = np.zeros(len(points))
    u12_errors = np.zeros(len(points))
    u21_errors = np.zeros(len(points))
    u22_errors = np.zeros(len(points))
    v11_errors = np.zeros(len(points))
    v12_errors = np.zeros(len(points))
    v21_errors = np.zeros(len(points))
    v22_errors = np.zeros(len(points))

    for i in range(len(points)):
        # sys.stdout("\r"+"{}/{} points evaluated".format(i, len(points)))
        point, antipoint = np.append(points[i], [1]), np.append(antipoints[i], [1])
        errors = []     # list to store all error components
        # project onto images of camera 1 and camera 2
        imgpoint1_cam1 = homogeneous2Inhomogeneous(p1 @ point)     # point 1 in first camera
        imgpoint2_cam1 = homogeneous2Inhomogeneous(p1 @ antipoint)     # point 2 in first camera
        imgpoint1_cam2 = homogeneous2Inhomogeneous(p2 @ point)     # point 1 in second camera
        imgpoint2_cam2 = homogeneous2Inhomogeneous(p2 @ antipoint)     # point 2 in second camera

        # imgpoint1_cam1 = imgpoint1_cam1[::-1]
        # imgpoint2_cam1 = imgpoint2_cam1[::-1]
        # imgpoint1_cam2 = imgpoint1_cam2[::-1]
        # imgpoint2_cam2 = imgpoint2_cam2[::-1]

        # now work out fractional error for each of these
        r = np.sum(np.sqrt((triangulate(imgpoint1_cam1, imgpoint1_cam2, p1, p2) - triangulate(imgpoint2_cam1, imgpoint2_cam2, p1, p2))**2))

        f1_errors[i] = alpha_r_f1 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1f, p2) - r
        f2_errors[i] = alpha_r_f2 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2f) - r
        x0_1_errors[i] = alpha_r_x0_1 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1x1, p2) - r
        y0_1_errors[i] = alpha_r_y0_1 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1y1, p2) - r
        x0_2_errors[i] = alpha_r_x0_2 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2x2) - r
        y0_2_errors[i] = alpha_r_y0_2 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2y2) - r
        yaw_errors[i] = alpha_r_yaw = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2_yaw) - r
        pitch_errors[i] = alpha_r_pitch = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2_pitch) - r
        roll_errors[i] = alpha_r_roll = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2_roll) - r
        xc_errors[i] = alpha_r_xc = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2cx) - r
        yc_errors[i] = alpha_r_yc = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2cy) - r
        zc_errors[i] = alpha_r_zc = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2cz) - r
        u11_errors[i] = alpha_r_u11 = relative_distance([imgpoint1_cam1[0]+u_err, imgpoint1_cam1[1]], imgpoint1_cam2, 
                                        imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r 
        u12_errors[i] = alpha_r_u12 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, [imgpoint2_cam1[0]+u_err, 
                                    imgpoint2_cam1[1]], imgpoint2_cam2, p1, p2) - r
        u21_errors[i] = alpha_r_u21 = relative_distance(imgpoint1_cam1, [imgpoint1_cam2[0]+u_err, imgpoint1_cam2[1]], 
                                        imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r
        u22_errors[i] = alpha_r_u22 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, [imgpoint2_cam2[0] + u_err, 
                                        imgpoint2_cam2[1]], p1, p2) - r
        v11_errors[i] = alpha_r_v11 = relative_distance([imgpoint1_cam1[0], imgpoint1_cam1[1]+v_err], imgpoint1_cam2, 
                                    imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r
        v12_errors[i] = alpha_r_v12 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, [imgpoint2_cam1[0], 
                                    imgpoint2_cam1[1]+v_err], imgpoint2_cam2, p1, p2) - r
        v21_errors[i] = alpha_r_v21 = relative_distance(imgpoint1_cam1, [imgpoint1_cam2[0], imgpoint1_cam2[1]+v_err], 
                                        imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r 
        v22_errors[i] = alpha_r_v22 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, [imgpoint2_cam2[0], 
                                        imgpoint2_cam2[1]+v_err], p1, p2) - r
        errors.append(alpha_r_f1)
        errors.append(alpha_r_f2)
        errors.append(alpha_r_x0_1)
        errors.append(alpha_r_y0_1)
        errors.append(alpha_r_x0_2)
        errors.append(alpha_r_y0_2)
        errors.append(alpha_r_yaw)
        errors.append(alpha_r_pitch)
        errors.append(alpha_r_roll)
        errors.append(alpha_r_xc)
        errors.append(alpha_r_yc)
        errors.append(alpha_r_zc)
        errors.append(alpha_r_u11)# image of point 1 x-coord in camera 1
        errors.append(alpha_r_v11) # image of point 1 y-coord in camera 1                      
        errors.append(alpha_r_u12)  # image of point 2 x-coord in camera 1
        errors.append(alpha_r_v12) # image of point 2 y-coord in camera 1                         
        errors.append(alpha_r_u21) # image of point 1 x-coord in camera 2
        errors.append(alpha_r_v21)# image of point 1 y-coord in camera 2
        errors.append(alpha_r_u22) # image of point 2 x-coord in camera 2
        errors.append(alpha_r_v22) # image of point 2 y-coord in v

        errors = np.asarray(errors)
        total_errors[i] = np.sqrt(np.sum(errors**2))

    print("maximal error for some orientation of two points is ", np.max(total_errors), "m")
    print("minimal error for some orientation of two points is ", np.min(total_errors), "m")


    print("Mean f1 error is {:.2g} m on relative distance".format(np.mean(f1_errors)))
    print("Mean f2 error is {:.2g} m on relative distance".format(np.mean(f2_errors)))
    print("Mean x0_1 error is {:.2g} m on relative distance".format(np.mean(x0_1_errors)))
    print("Mean x0_2 error is {:.2g} m on relative distance".format(np.mean(x0_2_errors)))
    print("Mean y0_1 error is {:.2g} m on relative distance".format(np.mean(y0_1_errors)))
    print("Mean y0_2 error is {:.2g} m on relative distance".format(np.mean(y0_2_errors)))
    print("Mean yaw error is {:.2g} m on relative distance".format(np.mean(yaw_errors)))
    print("Mean pitch error is {:.2g} m on relative distance".format(np.mean(pitch_errors)))
    print("Mean roll error is {:.2g} m on relative distance".format(np.mean(roll_errors)))
    print("Mean xc error is {:.2g} m on relative distance".format(np.mean(xc_errors)))
    print("Mean yc error is {:.2g} m on relative distance".format(np.mean(yc_errors)))
    print("Mean zc error is {:.2g} m on relative distance".format(np.mean(zc_errors)))
    print("Mean u11 error is {:.2g} m on relative distance".format(np.mean(u11_errors)))
    print("Mean u12 error is {:.2g} m on relative distance".format(np.mean(u12_errors)))
    print("Mean u21 error is {:.2g} m on relative distance".format(np.mean(u21_errors)))
    print("Mean u22 error is {:.2g} m on relative distance".format(np.mean(u22_errors)))
    print("Mean v11 error is {:.2g} m on relative distance".format(np.mean(v11_errors)))
    print("Mean v12 error is {:.2g} m on relative distance".format(np.mean(v12_errors)))
    print("Mean v21 error is {:.2g} m on relative distance".format(np.mean(v21_errors)))
    print("Mean v22 error is {:.2g} m on relative distance".format(np.mean(v22_errors)))




    return total_errors

class Camera:
    """
    Define Camera centre, focal length, sensor dimensions and pixel \
    dimensions...
    """

    def __init__(self, cameracentre, focal_length, sensor_width, sensor_height, pixel_size):
        self.centre = cameracentre
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.pixel_size = pixel_size

class Sim:
    """
    Simulation class, for the simulation of reconstruction algorithms. 
    
    Initialise with parameters: 
    ___________________________
    no_of_points : The number of points you want to randomly input into \
    the simulation.
    Cam1, Cam2 : Camera1 and Camera2 details, using the Cam class.
    yaw : The 3D yaw angle in degrees of camera 2 relative to camera1
    pitch : The 3D pitch angle in degrees of camera 2 relative to cam1
    roll : the 3D roll angle in degrees of camera 2 relative to camera1

    List of Functions:
    __________________
    # TODO
    """
    def __init__(self, no_of_points, Cam1, Cam2, yaw, pitch, roll, rad):
        self.tvec = Cam2.centre - Cam1.centre
        self.R = RotationMatrix(yaw, pitch, roll, rad=rad)

        points = np.zeros((no_of_points, 3))
        for i in range(points.shape[0]):
            points[i][0] = random.uniform(-100, 100)  # xs
            points[i][1] = random.uniform(-100, 100)  # ys
            points[i][2] = random.uniform(20, 100)  # zs

        self.points3D = points
        self.w1 = Cam1.sensor_width
        self.h1 = Cam1.sensor_height
        self.f1 = Cam1.focal_length
        self.p1 = Cam1.pixel_size
        self.w2 = Cam2.sensor_width
        self.h2 = Cam2.sensor_height
        self.f2 = Cam2.focal_length
        self.p2 = Cam2.pixel_size
        self.camera1centre = Cam1.centre
        self.camera2centre = Cam2.centre
        self.R = RotationMatrix(yaw, pitch, roll)
        self.tvec = Cam2.centre - Cam1.centre

        # define the sensor points in space:
        self.TL1 = np.array([-self.w1 / 2, self.h1 / 2, self.f1])  # Top Left 1
        self.TR1 = np.array([self.w1 / 2, self.h1 / 2, self.f1])  # Top Right 1
        self.BR1 = np.array([self.w1 / 2, -self.h1 / 2, self.f1])  # Bottom Right 1
        self.BL1 = np.array([-self.w1 / 2, -self.h1 / 2, self.f1])  # Bottom Left 1

        self.TL2 = transformpoint(np.array([-self.w2 / 2, self.h2 / 2, self.f2]), self.R, self.tvec)  # Top Left 2
        self.TR2 = transformpoint(np.array([self.w2 / 2, self.h2 / 2, self.f2]), self.R, self.tvec)  # Top Right 2
        self.BR2 = transformpoint(np.array([self.w2 / 2, -self.h2 / 2, self.f2]), self.R, self.tvec)  # Bottom Right 2
        self.BL2 = transformpoint(np.array([-self.w2 / 2, -self.h2 / 2, self.f2]), self.R, self.tvec)  # Bottom Left 2

    def testPoint(self, point):
        seen1, x1, y1 = pointInCamera1Image(point, self.w1, self.h1, self.f1, self.p1, self.camera1centre)
        seen2, x2, y2 = pointInCamera2Image(point, self.w2, self.h2, self.f2, self.p2, self.camera2centre, self.tvec,
                                            self.R)
        if seen1 and seen2:
            # fix the image coordinates
            x1, y1 = fixImgCoords(x1, y1, self.w1 / self.p1, self.h1 / self.p1)
            x2, y2 = fixImgCoords(x2, y2, self.w2 / self.p2, self.h2 / self.p2)

            stereoImages([x1], [y1], [x2], [y2], self.w1 / self.p1, self.w2 / self.p2, self.h1 / self.p1,
                         self.h2 / self.p2)

        else:
            print("Point not seen by BOTH cameras...")

    def synchImages(self):
        """Check if 3D point is visible to BOTH cameras. 
        \nIf so, then return it's pixel coordinates in each image... """
        x1s, y1s = [], []
        x2s, y2s = [], []
        seenpoints = []
        for point in self.points3D:
            seen1, x1, y1 = pointInCamera1Image(point, self.w1, self.h1, self.f1, self.p1, self.camera1centre)
            seen2, x2, y2 = pointInCamera2Image(point, self.w2, self.h2, self.f2, self.p2, self.camera2centre,
                                                self.tvec, self.R)
            # print((x1, y1))
            # print((x2, y1))

            if seen1 and seen2:
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)
                seenpoints.append(point)

        print("{} out of {} points can be seen in both images".format(len(x1s), len(self.points3D)))

        # fix the coordinates, so image origin is bottom left corner of CCD
        for i in range(len(x1s)):
            x1, y1 = fixImgCoords(x1s[i], y1s[i], self.w1 / self.p1, self.h1 / self.p1)
            x2, y2 = fixImgCoords(x2s[i], y2s[i], self.w2 / self.p2, self.h2 / self.p2)

            x1s[i] = x1
            y1s[i] = y1
            x2s[i] = x2
            y2s[i] = y2

        return x1s, y1s, x2s, y2s, seenpoints

    def seenpoints3D(self, seenpoints):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for point in seenpoints:
            x, y, z = point
            ax.plot([x, self.camera1centre[0]], [y, self.camera1centre[1]], [z, self.camera1centre[2]], c="m")
            ax.plot([x, self.camera2centre[0]], [y, self.camera2centre[1]], [z, self.camera2centre[2]], c="y")
            ax.scatter(x, y, z, c="b")

        ax.scatter(self.camera1centre[0], self.camera1centre[1], self.camera1centre[2], c="g", label="Camera 1 Centre")
        ax.scatter(self.camera2centre[0], self.camera2centre[1], self.camera2centre[2], c="r", label="Camera 2 Centre")

        ax.legend()
        ax.set_xlabel("x ")
        ax.set_ylabel("y ")
        ax.set_zlabel("z ")

    def drawImages(self, x1s, y1s, x2s, y2s):
        stereoImages(x1s, y1s, x2s, y2s, self.w1 / self.p1, self.w2 / self.p2, self.h1 / self.p1, self.h2 / self.p2)
        # plt.show()

    def scene3D(self):
        draw3D(self.points3D, self.camera1centre, self.camera2centre, self.TL1, self.TR1, self.BR1, self.BL1,
               self.TL2, self.TR2, self.BR2, self.BL2, self.tvec, self.R, shrink_factor=1)

    def returnPoints(self):
        return self.points3D

    def reconstruction(self):
        """
        Reconstruct the scene up to a projective ambiguity using image \
        correspondences.
        """
        x1s, y1s, x2s, y2s, seenpoints = Sim.synchImages(self)
        assert len(x1s) >= 8, "Cannot compute fundamental matrix with fewer than 8 point correspondences!"

        # convert image coordinates to correct form...
        img1coords, img2coords = convert_to_array(x1s, y1s, x2s, y2s)

        # derive camera matrices
        P1, P2, F = cameraMatrices(img1coords, img2coords)

        # TODO: USE THE OPTIMAL TRIANGULATION ALGORITHM!!!

        points_triangulated = optimal_triangulation(img1coords, img2coords, F)

        # Now we need to fix the projective ambiguity... 
        # Use ground control truth method...
        # TODO: This selection of ground truth points isn't working! 

        # Pick the ground control (euclidean) points, and their indexes within the 
        # reconstructed points
        gc_points, indexes = pickGroundTruthPoints(seenpoints, no_ground_truths=5)

        # make a new array to store the reconstructed versions of those 
        # euclidean known points... 
        rc_points = np.zeros((len(gc_points), 3))
        # print(rc_points)
        for i in range(len(indexes)):
            rc_points[i] = points_triangulated[indexes[i]]

        # print(rc_points, gc_points)

        P1, P2, newpoints = ground_truth_reconstruction(P1, P2, gc_points, rc_points, points_triangulated)


        return newpoints

def main():
    """# user input camera parameters:
    camcentre = input("Please Input Camera 2 Position (separated by commas): ")
    camera2centre = np.array([0,0,0])
    for i in range(len(camcentre.split(","))):
        cam2centre[i] = camcentre.split(",")[i]
    focal_length = float(input("Please input the focal lengths of the cameras (in metres): "))
    sensor_width = float(input("Input the width of the CCD (in metres): "))
    sensor_height= float(input("Input the height of the CCD (in metres): "))
    pixel_size   = float(input("Input the linear dimension of each pixel (in metres): "))"""

    cameracentre = np.array([0, 0, 0])  # let camera 1 lie at the origin of the coordinate system (m)
    sensor_width, sensor_height = 23.5e-3, 15.6e-3  # sensor dimensions of first camera (m)
    focal_length = 50e-3  # focal length of camera 1 (m)
    pixel_size = 3.9e-6  # linear dimension of a pixel (m)
    # point3D = np.array([[0, 0, 50]])
    camera2centre = np.array([25, 0, 0])

    Cam1 = Camera(cameracentre, focal_length, sensor_width, 
                  sensor_height, pixel_size)

    Cam2 = Camera(camera2centre, focal_length, sensor_width, 
                  sensor_height, pixel_size)

    sim = Sim(5000, Cam1, Cam2, yaw=-12, pitch=0, roll=0, rad=False)

    x1s, y1s, x2s, y2s, seenpoints = sim.synchImages()

    sim.drawImages(x1s, y1s, x2s, y2s)

    sim.scene3D()

    plt.show()

    ################################ IS MY ROTATION MATRIX WRONG???





    x1, x2 = convert_to_array(x1s, y1s, x2s, y2s)  # these are homogenous points

    print(x1.shape)

    A = formMatrixA(x1, x2)
    # print(A.shape)

    p1, p2, F = cameraMatrices(x1, x2)

    m = p2[:,-1]
    M = p2[:3,:3]

    print("Is F = [m]x M as it should? ", np.allclose(F, skew(m) @ M))
    print("F = ", F)
    print("[m]x M = ", skew(m) @ M)

    # print("p1 = ", p1)
    # print("p2 = ", p2)
    # print("F = ", F)

    f = F.reshape(9,1)
    print("Does Af = 0? ", np.allclose(A@f, 0))

    true = 0
    for i in range(len(x1)):
        if np.isclose(x2[i].T @ F @ x1[i], 0):
            true += 1
        else:
            pass
        
    print("x'Fx is true for {}/{} tests...".format(true, len(x1)))

    print("rank(F) = ", np.linalg.matrix_rank(F))
    print("det(F) = ", np.linalg.det(F))


    # points_triangulated = sim.reconstruction()
    # 3D DLT is Not working! This needs work...

    points_triangulated = optimal_triangulation(x1, x2, F)

    gc_points, indexes = pickGroundTruthPoints(seenpoints, no_ground_truths=6)
    rc_points = np.zeros((len(gc_points), 3))
    for i in range(len(indexes)):
            rc_points[i] = points_triangulated[indexes[i]]

    P1, P2, newpoints = ground_truth_reconstruction(p1, p2, gc_points, rc_points, points_triangulated)

    true1 = 0
    true2 = 0
    for i in range(len(x1)):
        X = newpoints[i]
        x_1, x_2 = x1[i], x2[i]
        print(homogeneous2Inhomogeneous(x_2)/homogeneous2Inhomogeneous(P2 @ X))
        if np.allclose(homogeneous2Inhomogeneous(x_1), homogeneous2Inhomogeneous(P1 @ X)):
            true1 += 1
        if np.allclose(homogeneous2Inhomogeneous(x_2), homogeneous2Inhomogeneous(P2 @ X)):
            true2 += 1


    # Correct using ground truths, or calibrated cameras? 

    # print("This should be skew-symmetric: \n", p2.T @ F @ p1)

    # check if x = PX
    # true1 = 0
    # true2 = 0
    # for i in range(len(x1)):
    #     X = points_triangulated[i]
    #     x_1, x_2 = x1[i], x2[i]
    #     if np.allclose(homogeneous2Inhomogeneous(x_1), homogeneous2Inhomogeneous(p1 @ np.append(X, [1]))):
    #         true1 += 1
    #     if np.allclose(homogeneous2Inhomogeneous(x_2), homogeneous2Inhomogeneous(p2 @ np.append(X, [1]))):
    #         true2 += 1
    
        # print("Are these equivalent?", homogeneous2Inhomogeneous(x_1), homogeneous2Inhomogeneous(p1 @ np.append(X, [1])))
        # print("Point mapped to image correctly in camera 1? ", np.allclose(homogeneous2Inhomogeneous(x_1), homogeneous2Inhomogeneous(p1 @ np.append(X, [1]))))
        # print("Point mapped correctly to image in camera 2? ", np.allclose(homogeneous2Inhomogeneous(x_2), homogeneous2Inhomogeneous(p2 @ np.append(X, [1]))))

    print("Points correctly mapped for {}/{} x=PX in camera 1".format(true1, len(x1)))
    print("Points correctly mapped for {}/{} x'=P'X in camera 2".format(true2, len(x1)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for point in newpoints:
        x, y, z = homogeneous2Inhomogeneous(point)

        ax.scatter(x, y, z)

    for point in seenpoints:
        x, y, z = point
        ax.scatter(x,y,z, c="k", alpha=0.2)

    # # create animated gif of reconstruction...
    # angles = np.linspace(0, 360, 26)[:-1]  # A list of 25 angles between 0 and 360
    # # create an animated .gif
    # for ii in range(0,360,10):
    #     ax.view_init(elev=10., azim=ii)
    #     plt.savefig("pics/movie{}.png".format(ii))


    # plt.show()

def test():
    # generate random camera matrices
    P1 = np.random.random(size=(3,4))
    P2 = np.random.random(size=(3,4))

    centre1 = np.array([[0,0,0]])
    centre2 = np.array([[50,0,0]])


    R2 = RotationMatrix(yaw=0, pitch=-45, roll=0)
    R1 = RotationMatrix(0,0,0)

    f = 50e-3 # in metres
    m = 6000/23.5e-3 # pixels per unit length.
    alpha = f*m  # focal length in pixels
    p_x, p_y = 3000, 2000 # centre pixels. 

    K = np.array([[alpha, 0, p_x], [0, alpha, p_y], [0,0,1]])

    p1 = K @ R1 @ np.c_[np.eye(3), centre1.reshape(3,1)]
    p2 = K @ R2 @ np.c_[np.eye(3), centre2.reshape(3,1)]

    print("Decompose camera matrix p1: K, R, C = \n", decomposeCameraMtx(p1))
    print("Decompose camera matrix p2: K, R, C = \n", decomposeCameraMtx(p2))

    Cam1 = Camera(centre1, f, 23.5e-3, 
                  15.6e-3, 6000/23.5e-3)

    Cam2 = Camera(centre2, f, 23.5e-3, 
                  15.6e-3, 6000/23.5e-3)

    sim = Sim(300, Cam1, Cam2, yaw=0, pitch=-45, roll=0, rad=False)

    x1s, y1s, x2s, y2s, seenpoints = sim.synchImages()

    sim.drawImages(x1s, y1s, x2s, y2s)

    sim.scene3D()

    plt.show()

    coords1, coords2 = convert_to_array(x1s, y1s, x2s, y2s)

    # generate random 3D points
    points = np.zeros((300, 4))
    for i in range(points.shape[0]):
        points[i][0] = random.uniform(-10, 10)  # xs
        points[i][1] = random.uniform(-10, 10)  # ys
        points[i][2] = random.uniform(45, 55)   # zs
        points[i][3] = 1 # make homogeneous

    # project to image points on each camera
    x1, x2 = np.zeros((points.shape[0], 2)), np.zeros((points.shape[0], 2))
    x1s, y1s, x2s, y2s = [], [], [], []
    for i in range(len(points)):
        x1[i] = homogeneous2Inhomogeneous(p1 @ points[i])
        x2[i] = homogeneous2Inhomogeneous(p2 @ points[i])
        x1s.append(x1[i][0])
        y1s.append(x1[i][1])
        x2s.append(x2[i][0])
        y2s.append(x2[i][1])

    img1coords, img2coords = convert_to_array(x1s, y1s, x2s, y2s)

    print(coords1, coords2)
    print(img1coords, img2coords)


    true = 0
    true2 = 0
    triangulated_points = np.zeros((points.shape[0], 3))
    for i in range(len(points)):
        triangulated_points[i][:] = triangulate(img1coords[i], img2coords[i], p1, p2)
        if np.allclose(triangulated_points[i], points[i][:3]):
            true2 += 1
        A = formA(x1[i], x2[i], p1, p2)
        # print(A @ points[i])
        if np.allclose(A @ points[i], 0):
            true += 1
        else:
            pass

    # print(points[2], triangulated_points[2])

    print("AX=0 for {}/{} points".format(true, len(points)))
    print("{}/{} points were correctly triangulated".format(true2, len(points)))

    p1, p2, F = cameraMatrices(img1coords, img2coords)

    test1 = 0
    for i in range(len(img1coords)):
        if np.allclose(img2coords[i].T @ F @ img1coords[i], 0):
            test1 += 1

    print("Epipolar constraint x'Fx=0 satisfied for {}/{} points".format(test1, len(img1coords)))

    # T1 = transform_2d(x1)
    # T2 = transform_2d(x2)

    # test4 = 0 
    # for i in range(len(x1)):
    #     A = formA(x1[i], x2[i], P1, P2)
    #     normpoint = right_null_space(A).reshape(4,1)
    #     d = np.linalg.svd(A, compute_uv=0)
    #     # print(d)

    #     if np.allclose(A @ normpoint, 0):
    #         test4 += 1

    # print("Normalised AX=0 true for {}/{} points".format(test4, len(x1)))

    # test2 = 0
    # for i in range(len(points)):
    #     A = formA(x1[i], x2[i], p1, p2)
    #     # print(A @ points[i])
    #     if np.allclose(A @ points[i], 0):
    #         test2 += 1
    #     else:
    #         pass
    # print("Does AX=0 for determined P1 and P2? True for {}/{} points".format(test2, len(points)))

    # _, _, C = decomposeCameraMtx(P1)
    # e2 = P2 @ np.append(C, [1])

    # test_F = skew(e2) @ P2 @ np.linalg.pinv(P1)

    # sf = test_F / F  # defined F up to a scale
    # # print("Scale factor of F is ", sf)
    # test3 = 0
    # for i in range(len(img1coords)):
    #     if np.allclose(img2coords[i].T @ test_F @ img1coords[i], 0):
    #         test3 += 1

    # print("Epipolar constraint for F determined from camera matrices is satisfied for {}/{} points".format(test3, len(img1coords)))

    # print("Is F what it should be? ", np.allclose(F, test_F/sf[0,0]))
    # # print(F, test_F/sf[0,0])


    # # p1 = [I|0], p2 = [M|m] then F = skew(m)M
    # m = p2[:,-1]
    # M = p2[:3, :3]
    # # print(p2, M, m)
    
    # print("F = {}, same as \n{}?".format(F, (skew(m)@M)))
    # print(np.isclose(F, -(skew(m)@M)))

    return None

def test2():
    centre1 = np.array([[0,0,0]])
    centre2 = np.array([[20,0,0]])
    # centre2 = np.array([[200, -10, 200]])

    R2 = RotationMatrix(yaw=0, pitch=-10, roll=0)
    R1 = RotationMatrix(0,0,0)
    # print("R = ", RotationMatrix(0, -90, 0))
    # R2 = RotationMatrix(0, -90, 0)

    f = 50e-3 # in metres
    m = 6000/23.5e-3 # pixels per unit length.
    alpha = f*m  # focal length in pixels
    p_x, p_y = 3000, 2000 # centre pixels. 

    K = np.array([[alpha, 0, p_x], [0, alpha, p_y], [0,0,1]])

    p1 = K @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]

    # generate random 3D points, must be in-front of camera, no ability to tell if point is infront! 
    points = np.zeros((200, 4))
    for i in range(points.shape[0]):
        points[i][0] = random.uniform(0, 50)        # xs
        points[i][1] = random.uniform(0, 5)         # ys
        points[i][2] = random.uniform(100, 150)     # zs
        points[i][3] = 1 # make homogeneous

    # print("The point X = (5, -10, 200) imaged? ")
    # print(homogeneous2Inhomogeneous(p2 @ np.array([5, -10, 200, 1])))

    # print(homogeneous2Inhomogeneous(p1 @ np.array([30,0,10000, 1])))

    # project to image points on each camera
    x1, x2 = np.zeros((points.shape[0], 2)), np.zeros((points.shape[0], 2))
    x1s, y1s, x2s, y2s = [], [], [], []
    for i in range(len(points)):
        x1[i] = homogeneous2Inhomogeneous(p1 @ points[i])
        x2[i] = homogeneous2Inhomogeneous(p2 @ points[i])
        x1s.append(x1[i][0])
        y1s.append(x1[i][1])
        x2s.append(x2[i][0])
        y2s.append(x2[i][1])

    img1coords, img2coords = convert_to_array(x1s, y1s, x2s, y2s)
    # print(img1coords)


    true = 0
    true2 = 0
    rms = []
    triangulated_points = np.zeros((points.shape[0], 3))
    for i in range(len(points)):
        triangulated_points[i][:] = triangulate(img1coords[i], img2coords[i], p1, p2)
        difference = triangulated_points[i] - points[i][:3]

        rms.append(np.sqrt(np.sum(difference**2)))
        if np.allclose(triangulated_points[i], points[i][:3]) and points[i][3] == 1:
            true2 += 1
        A = formA(img1coords[i][:2], img2coords[i][:2], p1, p2)
        # print(A @ points[i])
        if np.allclose(A @ points[i], 0):
            true += 1
        else:
            pass

    print("AX=0 for {}/{} points".format(true, len(points)))
    print("{}/{} points were correctly triangulated".format(true2, len(points)))
    print("average rms error in triangulation: {} +- units".format(np.mean(rms)))

    # Now let's sort out which images both cameras can actually see! 
    seenx1s, seenx2s, seeny1s, seeny2s = [], [], [], []
    for i in range(len(img1coords)):
        xx1, yy1, _ = img1coords[i]
        xx2, yy2, _ = img2coords[i]


        seenx, seeny = False, False
        if (xx1 >= 0) and (xx1 <= 6000):
            if (xx2 >= 0) and (xx2 <= 6000):
                seenx = True
            else:
                pass
        if (yy1 >= 0) and (yy1 <= 4000):
            if (yy2 >= 0) and (yy2 <= 4000):
                seeny = True
            else:
                pass
        else:
            pass

        # if seenx and seeny:
        if seenx and seeny:
            seenx1s.append(xx1)
            seenx2s.append(xx2)
            seeny1s.append(yy1)
            seeny2s.append(yy2)

    # print(len(seenx1s), len(seeny1s))
    seenimg1coords, seenimg2coords = convert_to_array(seenx1s, seeny2s, seenx2s, seeny2s) 

        # plot the imaged points
    fig = plt.figure()
    ax1 = plt.subplot(121)
    plt.plot(x1s, y1s, "bx")
    plt.plot(seenx1s, seeny1s, "rx")
    # plot the sensor on
    w1 = w2 = 23.5e-3/ (23.5e-3/6000)
    h1 = h2 = 15.6e-3 / (15.6e-3/4000)


    plt.plot([0, w1], [0, 0], "r", alpha=0.3)  # bottom horizontal
    plt.plot([w1, w1], [0, h1], "r", alpha=0.3)  # right vertical
    plt.plot([w1, 0], [h1, h1], "r", alpha=0.3)  # top horizontal
    plt.plot([0, 0], [h1, 0], "r", alpha=0.3)  # left vertical

    ax1.set_aspect("equal")
    plt.xlabel("x-direction (pixels)")
    plt.ylabel("y-direction (pixels)")
    plt.title("Camera 1 Image")
    plt.ylim(-100, 4100)
    plt.xlim(-100, 6100)

    ax2 = plt.subplot(122, sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.plot(x2s, y2s, "bx")
    plt.plot(seenx2s, seeny2s, "rx")

    # plot the sensor on
    plt.plot([0, w2], [0, 0], "r", alpha=0.3)
    plt.plot([w2, w2], [0, h2], "r", alpha=0.3)
    plt.plot([w2, 0], [h2, h2], "r", alpha=0.3)
    plt.plot([0, 0], [h2, 0], "r", alpha=0.3)

    ax2.set_aspect("equal")
    plt.title("Camera 2 Image")
    plt.xlabel("x-direction (pixels)")
    plt.ylim(-100, 4100)
    plt.xlim(-100, 6100)

    plt.suptitle("Points as Imaged by Two Stereo-Cameras")

    plt.show()

    # print(seenimg2coords)
    print(len(seenimg2coords), " points are seen in both images")
    P1, P2, F = cameraMatrices(img1coords, img2coords)

    

    e2 = right_null_space(F.T)

    # e2 = P'C
    # e2 = (p2 @ np.append(centre1, [1]))
    # print(e2)

    # print(homogeneous2Inhomogeneous(sp.linalg.null_space(p2)))
    
    # F = skew(e2) @ p2 @ np.linalg.pinv(p1)

    # print("F has rank ", np.linalg.matrix_rank(F))

    M = P2[:3,:3]
    m = P2[:,-1]
    
    

    test1 = 0
    for i in range(len(img1coords)):
        if np.allclose(img2coords[i].T @ F @ img1coords[i], 0):
            test1 += 1

    print("Epipolar constraint x'Fx=0 satisfied for {}/{} points".format(test1, len(img1coords)))

    # print(homogeneous2Inhomogeneous(P1 @ points[3]), homogeneous2Inhomogeneous(p1@points[3]))

    # try using essential matrix...
    K1, _, _ = decomposeCameraMtx(p1)
    K2, _, _ = decomposeCameraMtx(p2)

    E = K2.T @ F @ K1

    d = np.linalg.svd(E, compute_uv=False)
    # print("Singular values of E are : ", d)

    P1, P2 = findCamerafromEssentialMTX(E, points[33])
    print(P1, P2)
    print(p1, p2)

    # check if x'Ex = 0 
    test5 = 0
    for i in range(len(img1coords)):
        if np.allclose(np.linalg.inv(K2)@img2coords[i] @ E @ np.linalg.inv(K1)@img1coords[i], 0):
            test5 += 1
        else:
            pass

    print("x'Ex = 0 true for {}/{} points".format(test5,len(img1coords)))

    test6 = 0
    test7 = 0
    for i in range(len(points)):
        img1 = homogeneous2Inhomogeneous(P1 @ points[i])
        # print(img1/homogeneous2Inhomogeneous(img1coords[i]))
        img2 = homogeneous2Inhomogeneous(P2 @ points[i])
        if np.allclose(img1, homogeneous2Inhomogeneous(img1coords[i])):
            test6 += 1
        if np.allclose(img2, homogeneous2Inhomogeneous(img2coords[i])):
            test7 += 1
        else:
            pass

    print("P1 from essential matrix satisfies {}/{} points for camera 1".format(test6, len(img1coords)))
    print("P2 from essential matrix satisfies {}/{} points for camera 2".format(test7, len(img1coords)))



    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for point in points:
        x, y, z = point[:3]
        ax.scatter(x,y,z, color="r")
    
    for i in range(len(seenimg1coords)):
        x, y, z = triangulate(seenimg1coords[i], seenimg2coords[i], p1, p2)
        ax.scatter(x, y, z, color="g")
        ax.plot((x,0), (y,0), (z,0), color="y")
        ax.plot((x,centre2[0][0]), (y,centre2[0][1]), (z,centre1[0][2]), color="orange")

    ax.scatter(0,0,0, color="c", label="Camera centres")
    ax.scatter(centre2[0][0], centre2[0][1], centre1[0][2], color="c")

    ax.legend()
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")

    # plt.show()

def final():
    # use camera matrices P1 and P2 to generate image points for 
    # some number of 3D points. 
    # From image correspondences (just imagine sensor is infinitely large)
    # then compute F and choose P1 and P2. 
    # Triangulate and check that AX=0 from formed P1 and P2. 
    # Then use DLT to fix projective ambiguity. 

    centre1 = np.array([[0,0,0]])   # Camera 1 centre (origin)
    centre2 = np.array([[50,0,0]])  # Camera 2 centre
    R2 = RotationMatrix(yaw=-10, pitch=0, roll=0)   # Camera 2 orientation (compared to camera 1)
    R1 = RotationMatrix(0,0,0)                      # Camera 1 orientation (none! )
    f = 50e-3                                       # Focal Length (m)
    m = 6000/23.5e-3                                # Pixels per unit length
    alpha = f*m                                     # Focal length in pixels   
    p_x, p_y = 3000, 2000                           # Centre pixels. (Prinicpal point)
    f_error = 1e-3*m                                # Focal length uncertainty (pixels)
    x_error = 5                                     # Uncertainty in position of principal point
    y_error = 5                                     # Uncertainty in position of principal point

    K = np.array([[alpha, 0, p_x], [0, alpha, p_y], [0,0,1]])   # Calibration matrix (same for both cameras...)

    p1 = K @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]       # Camera matrix 1
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]       # Camera matrix 2

    # generate random 3D points, must be in-front of camera, no ability to tell if point is infront! 
    points = np.zeros((200, 4))
    for i in range(points.shape[0]):
        points[i][0] = random.uniform(0, 50)        # xs
        points[i][1] = random.uniform(0, 5)         # ys
        points[i][2] = random.uniform(100, 150)     # zs
        points[i][3] = 1 # make homogeneous

    # Map onto 2D image coordinates (We assume images are infinitely sized, so we can see all points...) 
    x1, x2 = np.zeros((points.shape[0], 2)), np.zeros((points.shape[0], 2)) 
    x1s, y1s, x2s, y2s = [], [], [], []
    for i in range(len(points)):
        x1[i] = homogeneous2Inhomogeneous(p1 @ points[i])
        x2[i] = homogeneous2Inhomogeneous(p2 @ points[i])
        x1s.append(x1[i][0])
        y1s.append(x1[i][1])
        x2s.append(x2[i][0])
        y2s.append(x2[i][1])

    img1coords, img2coords = convert_to_array(x1s, y1s, x2s, y2s)  # np.array of image points (homogeneous coordinates)

    P1, P2, F = cameraMatrices(img1coords, img2coords)

    # test if x'Fx = 0
    test = 0
    for i in range(len(img1coords)):
        x1, x2 = img1coords[i], img2coords[i]
        if np.allclose(x2.T @ F @ x1, 0):
            test += 1
        else:
            pass

    print("x'Fx = 0 for {}/{} points".format(test, len(img1coords)))


    # triangulated_points = optimal_triangulation(img1coords, img2coords, F)


    # testing "sphere of results" by monte carlo simulation on single point
    point = np.array([10, 0, 100, 1])
    imgpoint1 = homogeneous2Inhomogeneous(p1 @ point) 
    imgpoint2 = homogeneous2Inhomogeneous(p2 @ point) 


    triangulated_point = triangulate(imgpoint1, imgpoint2, p1, p2)


    # FUNCTIONAL APPROACH
    mean_X = np.asarray(triangulated_point)   # best guess 3D point
    # so parameters that can vary (as measured) are focal lengths, principal points (x,y), camera centre (x,y,z) of camera 2,
    # rotation matrix of camera 2 (alpha, beta, gamma angles). 
    # f1, f2, x1, x2, y1, y2, x_c, y_c, z_c, alpha, beta, gamma. We take camera 1 as to be exactly at the origin, by definition,
    # and any measurements of rotation are relative to camera 1. 
    # 
    # How to turn these, and any uncertainties into uncertainty?? 
    # How to do covariance matrix?

    f_error = 5e-3*m
    x_error = 1
    y_error = 1
    alpha_err = np.rad2deg(0.0004)          # Error in yaw angle (from STARFLAG)
    beta_err  = np.rad2deg(0.0004)#np.rad2deg(0.02)            # Error in pitch angle (from STARFLAG)
    gamma_err = np.rad2deg(0.0004)          # Error in roll angle (from STARFLAG)

    f1_err = f2_err = f_err = f_error       # define as all equal for sake of simulation
    x1_err = x2_err = x_error               # principal point x error
    y1_err = y2_err = y_error               # principal point y error
    xc_err = yc_err = zc_err = 0.01         # error in camera 2 centre coordinates ~(1 cm)
    imgcoordx_err = 3                       # Error in measured point in x-direction (pixels)
    imgcoordy_err = 3                       # Error in measured point in y-direction (pixels)

    yaw, pitch, roll = -10, 0, 0

    # K[0,0] = f = K[1,1]
    # K[0,2] = p_x
    # K[1,2] = p_y
    # centre2 = np.array([[20,0,0]])
    # R2 = RotationMatrix(yaw=0, pitch=-10, roll=0) 
    # P = K @ R @ np.c_[np.eye(3), -centre.reshape(3,1)]

    p1f = np.array([[alpha+f_err, 0, p_x], [0,alpha+f_err, p_y], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2f = np.array([[alpha+f_err, 0, p_x], [0,alpha+f_err, p_y], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1x1 = np.array([[alpha, 0, p_x+x1_err], [0,alpha, p_y], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2x2 = np.array([[alpha, 0, p_x+x2_err], [0,alpha, p_y], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1y1 = np.array([[alpha, 0, p_x], [0,alpha, p_y+y1_err], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2y2 = np.array([[alpha, 0, p_x], [0,alpha, p_y+y2_err], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2cx = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0][0] + xc_err], [centre2[0][1]], [centre2[0][2]]])]
    p2cy = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0][0]], [centre2[0][1]+yc_err], [centre2[0][2]]])]
    p2cz = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0][0]], [centre2[0][1]], [centre2[0][2]+zc_err]])]
    p2_yaw = K @ RotationMatrix(yaw+alpha_err, pitch, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_pitch = K @ RotationMatrix(yaw, pitch+beta_err, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_roll = K @ RotationMatrix(yaw, pitch, roll+gamma_err) @ np.c_[np.eye(3), -centre2.reshape(3,1)]


    alpha_f1 = np.asarray(triangulate(imgpoint1, imgpoint2, p1f, p2) ) - mean_X
    alpha_y1 = np.asarray(triangulate(imgpoint1, imgpoint2, p1y1, p2) ) - mean_X
    alpha_f2 = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2f) ) - mean_X
    alpha_x1 = np.asarray(triangulate(imgpoint1, imgpoint2, p1x1, p2) ) - mean_X
    alpha_x2 = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2x2) ) - mean_X
    alpha_y2 = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2y2) ) - mean_X
    alpha_xc = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2cx)  ) - mean_X
    alpha_yc = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2cy) ) - mean_X
    alpha_zc = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2cz) ) - mean_X
    alpha_alpha = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2_yaw)) - mean_X
    alpha_beta = np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2_pitch) ) - mean_X
    alpha_gamma= np.asarray(triangulate(imgpoint1, imgpoint2, p1, p2_roll) ) - mean_X
    alpha_img1x = np.asarray(triangulate([imgpoint1[0]+imgcoordx_err, imgpoint1[1]], imgpoint2, p1, p2) ) - mean_X
    alpha_img1y = np.asarray(triangulate([imgpoint1[0], imgpoint1[1]+imgcoordy_err], imgpoint2, p1, p2) ) - mean_X
    alpha_img2x = np.asarray(triangulate(imgpoint1, [imgpoint2[0]+imgcoordx_err, imgpoint2[1]], p1, p2) ) - mean_X
    alpha_img2y = np.asarray(triangulate(imgpoint1, [imgpoint2[0], imgpoint2[1]+imgcoordy_err], p1, p2) ) - mean_X

    total_err = np.sqrt(alpha_f1**2 + alpha_f2**2 + alpha_x1**2 + alpha_x2**2 + alpha_y1**2 + alpha_y2**2 +  
                alpha_xc**2 + alpha_yc**2 + alpha_zc**2 + alpha_alpha**2 + alpha_beta**2 + alpha_gamma**2 +
                alpha_img1x**2 + alpha_img1y**2 + alpha_img2x**2 + alpha_img2y**2)
    print(mean_X, " +- ", total_err)

    # print(triangulated_point)
    print("error from f1 ", alpha_f1)
    print('error from py_1 ', alpha_y1)
    print('error from f2', alpha_f2)
    print('error from px_1 ', alpha_x1)
    print('error from px_2 ', alpha_x2)
    print('error from py_2 ', alpha_y2)
    print('error from camera x', alpha_xc)
    print('error from camera y ', alpha_yc)
    print('error from camera z ', alpha_zc)
    print('error from yaw ', alpha_alpha)
    print('error from pitch ', alpha_beta)
    print('error from roll ', alpha_gamma)
    print('error from u1 ', alpha_img1x)
    print('error from v1 ', alpha_img1y)
    print('error from u2 ', alpha_img2x)
    print('error from v2 ', alpha_img2y)

    functional_error(alpha, f_err, p_x, x_error, p_y, y_error, centre2[0][0], xc_err, centre2[0][1], yc_err, centre2[0][2], zc_err, yaw, alpha_err, pitch, beta_err, roll, gamma_err, imgpoint1[0], imgpoint2[0], imgcoordx_err, imgpoint1[1], imgpoint2[1], imgcoordy_err)
    
    vary_coordz = np.linspace(10, 110, 30)
    vary_coordxy = np.linspace(-50, 70, len(vary_coordz))
    errorsz = np.zeros((len(vary_coordz), 3))
    errorsx = np.zeros(errorsz.shape)
    errorsy = np.zeros(errorsz.shape)

    for i in range(len(vary_coordz)):
        print("{}/{} complete...".format(i+1, len(vary_coordz)), flush=True)
        errorsz[i] = nearest_neighbour_err(alpha, f_err, p_x, x_error, p_y, y_error, centre2[0][0], xc_err, 
                                        centre2[0][1], yc_err, centre2[0][2], zc_err, yaw, alpha_err, pitch, beta_err, 
                                        roll, gamma_err, imgcoordx_err, imgcoordy_err, 
                                        np.array([10, 0, vary_coordz[i]]), 1000)
        errorsx[i] = nearest_neighbour_err(alpha, f_err, p_x, x_error, p_y, y_error, centre2[0][0], xc_err, 
                                        centre2[0][1], yc_err, centre2[0][2], zc_err, yaw, alpha_err, pitch, beta_err, 
                                        roll, gamma_err, imgcoordx_err, imgcoordy_err, 
                                        np.array([vary_coordxy[i], 0, 25]), 1000)
        errorsy[i] = nearest_neighbour_err(alpha, f_err, p_x, x_error, p_y, y_error, centre2[0][0], xc_err, 
                                        centre2[0][1], yc_err, centre2[0][2], zc_err, yaw, alpha_err, pitch, beta_err, 
                                        roll, gamma_err, imgcoordx_err, imgcoordy_err, 
                                        np.array([0, vary_coordxy[i], 25]), 1000)

    # print(errors)

    # plot graph of the error in the mean error in each coordinate with depth z
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs[2].plot(vary_coordz, errorsz[:,0], "bx", ms=4)
    axs[1].plot(vary_coordz, errorsz[:,1], "bx", ms=4)
    axs[0].plot(vary_coordz, errorsz[:,2], "bx", ms=4)

    axs[2].set_ylabel("x-error (m)")
    axs[1].set_ylabel("y-error (m)")
    axs[0].set_ylabel("z-error (m)")

    axs[2].set_xlabel("depth (m)")
    fig.tight_layout()

    plt.suptitle("Mean coordinate error over many orientations at varying depth z")

    # plot error with varying x-coord
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs[2].plot(vary_coordxy, errorsx[:,0], "bx", ms=4)
    axs[1].plot(vary_coordxy, errorsx[:,1], "bx", ms=4)
    axs[0].plot(vary_coordxy, errorsx[:,2], "bx", ms=4)

    axs[2].set_ylabel("x-error (m)")
    axs[1].set_ylabel("y-error (m)")
    axs[0].set_ylabel("z-error (m)")

    axs[2].set_xlabel("x-position (m)")

    fig.tight_layout()

    plt.suptitle("Mean coordinate error over many orientations at varying x-position")

    # plot error with varying y-coord
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs[2].plot(vary_coordxy, errorsy[:,0], "bx", ms=4)
    axs[1].plot(vary_coordxy, errorsy[:,1], "bx", ms=4)
    axs[0].plot(vary_coordxy, errorsy[:,2], "bx", ms=4)

    axs[2].set_ylabel("x-error (m)")
    axs[1].set_ylabel("y-error (m)")
    axs[0].set_ylabel("z-error (m)")

    axs[2].set_xlabel("y-position (m)")

    plt.suptitle("Mean coordinate error over many orientations at varying y-position")
    fig.tight_layout()
    plt.show()




    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point[0], point[1], point[2], color="g")
    for i in range(2000):
        p1 = perturb_camera_matrix(K, R1, centre1, f_error, x_error, y_error)
        p2 = perturb_camera_matrix(K, R2, centre2, f_error, x_error, y_error)
        triangulated_point = triangulate(imgpoint1, imgpoint2, p1, p2)
        x, y, z = triangulated_point
        ax.scatter(x, y, z, color="r")
    

    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")
    plt.show()

    # print(point, triangulated_point)"""

def fov():
    centre1 = np.array([0,0,0])   # Camera 1 centre (origin)
    centre2 = np.array([50,0,0])  # Camera 2 centre
    yaw, pitch, roll = 0, 0, 0
    R2 = RotationMatrix(yaw=yaw, pitch=pitch, roll=roll)   # Camera 2 orientation (compared to camera 1)
    R1 = RotationMatrix(0,0,0)                      # Camera 1 orientation (none! )
    f = 50e-3                                       # Focal Length (m)
    m = 6000/23.5e-3                                # Pixels per unit length
    alpha = f*m                                     # Focal length in pixels   
    p_x, p_y = 3000, 2000                           # Centre pixels. (Prinicpal point)
    f_error = 1e-3*m                                # Focal length uncertainty (pixels)
    x_error = 5                                     # Uncertainty in position of principal point
    y_error = 5                                     # Uncertainty in position of principal point

    K = np.array([[alpha, 0, p_x], [0, alpha, p_y], [0,0,1]])   # Calibration matrix (same for both cameras...)

    p1 = K @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]       # Camera matrix 1
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]       # Camera matrix 2

    f_error = 1e-3*m
    x_error = 1
    y_error = 1
    alpha_err = 0.1#np.rad2deg(0.0004)         # Error in yaw angle (from STARFLAG)
    beta_err  = .1#np.rad2deg(0.02)            # Error in pitch angle (from STARFLAG)
    gamma_err = .1#np.rad2deg(0.0004)          # Error in roll angle (from STARFLAG)

    f1_err = f2_err = f_err = f_error       # define as all equal for sake of simulation
    x1_err = x2_err = x_error               # principal point x error
    y1_err = y2_err = y_error               # principal point y error
    xc_err = yc_err = zc_err = 0.01         # error in camera 2 centre coordinates ~(1 cm)
    imgcoordx_err = 3                       # Error in measured point in x-direction (pixels)
    imgcoordy_err = 3                       # Error in measured point in y-direction (pixels)

    

    # vary x,y over all combinations of coordinates for -50 to 50.
    vals = np.linspace(-50, 70, 121)

    xs = np.zeros(len(vals)**2)
    ys = np.zeros(len(vals)**2)
    counter = 0
    for i in range(len(vals)):
        for j in range(len(vals)):
            xs[counter] = vals[i]
            ys[counter] = vals[j]
            counter += 1

    # for each of these x,y,z coordinates reconstruct the point using the functional approach, and return the error in x,y,z

    total_err = np.zeros(xs.shape)
    xyz_errs = np.zeros((len(xs), 3))
    for i, (x, y) in enumerate(zip(xs, ys)):
        point = np.array([x,y,50,1])
        img1coord = homogeneous2Inhomogeneous(p1 @ point)
        img2coord = homogeneous2Inhomogeneous(p2 @ point)

        mean, err = functional_error(alpha, f_err, p_x, x_error, p_y, y_error, centre2[0], xc_err, centre2[1], yc_err,
                                    centre2[2], zc_err, yaw, alpha_err, pitch, beta_err, roll, gamma_err, img1coord[0], 
                                    img2coord[0], imgcoordx_err, img1coord[1], img2coord[1], imgcoordy_err)

        # print(point, mean)
        # print(mean, err)
        # print(np.sqrt(err[0]**2 + err[1]**2 + err[2]**2))
        xyz_errs[i] = err
        total_err[i] = np.sqrt(err[0]**2 + err[1]**2 + err[2]**2)
    # print(xs)
    # print(ys)
    # print(total_err)

    scaled_z = (total_err - total_err.min()) / total_err.ptp()
    colors = plt.cm.coolwarm(scaled_z)

    cvals  = [-2., -1, 2]

    colors = ["green","orange","red"]

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    fig = plt.figure(111)
    plt.scatter(xs, ys, c=total_err, marker="o", cmap=cmap, s = 20, linewidths=4)
    cbar = plt.colorbar()

    plt.xlabel("x-position (m)")
    plt.ylabel("y-position (m)")
    cbar.set_label('Absolute distance uncertainty (m)', rotation=270, labelpad=20)
    plt.suptitle(r"Total Uncertainty for varying $x,y$ position at fixed $z$=50 m")
    plt.savefig('{}_convergence_angle.png'.format(yaw))

    # fig.tight_layout()
    plt.show()

def rel_dist():
    # so relative distance between two points X1 and X2 is r = sqrt((X2-X1)^2)
    # to do functional approach, need r as a function of f, x0, y0, yaw, pitch roll etc... 
    # Then can vary parameters as required. Obviously, both images are taken simultaneously, 
    # so camera intrinsic parameters and orientation is fixed uncertainty for BOTH measurements, 
    # but the measured positions of those points can vary independently. 

    # i.e for the u,v image uncertainty, we need to vary u, v independently from each other
    # and independently for both objects in the scene. 

    centre1 = np.array([0,0,0])   # Camera 1 centre (origin)
    centre2 = np.array([25,0,0])  # Camera 2 centre
    yaw, pitch, roll = -12, 0, 0
    R2 = RotationMatrix(yaw=yaw, pitch=pitch, roll=roll)   # Camera 2 orientation (compared to camera 1)
    R1 = np.eye(3)                                # Camera 1 orientation (none! )
    f = 50e-3                                       # Focal Length (m)
    m = 6000/23.5e-3                                # Pixels per unit length
    alpha = f*m                                     # Focal length in pixels   
    p_x, p_y = 3000, 2000                           # Centre pixels. (Prinicpal point)
    f_error = 1e-3*m                                # Focal length uncertainty (pixels)
    x_error = 5                                     # Uncertainty in position of principal point
    y_error = 5                                     # Uncertainty in position of principal point

    K = np.array([[alpha, 0, p_x], [0, alpha, p_y], [0,0,1]])   # Calibration matrix (same for both cameras...)

    p1 = K @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]       # Camera matrix 1
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]       # Camera matrix 2

    f_error = 1e-3*m
    x_error = 1
    y_error = 1
    alpha_err = np.rad2deg(0.0004)          # Error in yaw angle (from STARFLAG)
    beta_err  = np.rad2deg(0.02)#np.rad2deg(0.02)            # Error in pitch angle (from STARFLAG)
    gamma_err = np.rad2deg(0.0004)          # Error in roll angle (from STARFLAG)

    f1_err = f2_err = f_err = f_error       # define as all equal for sake of simulation
    x1_err = x2_err = x_error               # principal point x error
    y1_err = y2_err = y_error               # principal point y error
    xc_err = yc_err = zc_err = 0.01         # error in camera 2 centre coordinates ~(1 cm)
    imgcoordx_err = 3                    # Error in measured point in x-direction (pixels)
    imgcoordy_err = 3                    # Error in measured point in y-direction (pixels)

    

    # point1, point2 = np.array([14, 0, 30, 1]), np.array([16, 0, 30, 1])  # separated by KNOWN 2M
    point1, point2 = np.array([14, 0, 30, 1]), np.array([14, 0, 32, 1])  # separated by KNOWN 2M

    imgpoint1_cam1 = homogeneous2Inhomogeneous(p1 @ point1)     # point 1 in first camera
    imgpoint2_cam1 = homogeneous2Inhomogeneous(p1 @ point2)     # point 2 in first camera
    imgpoint1_cam2 = homogeneous2Inhomogeneous(p2 @ point1)     # point 1 in second camera
    imgpoint2_cam2 = homogeneous2Inhomogeneous(p2 @ point2)     # point 2 in second camera

    p1f = np.array([[alpha+f_err, 0, p_x], [0,alpha+f_err, p_y], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2f = np.array([[alpha+f_err, 0, p_x], [0,alpha+f_err, p_y], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1x1 = np.array([[alpha, 0, p_x+x1_err], [0,alpha, p_y], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2x2 = np.array([[alpha, 0, p_x+x2_err], [0,alpha, p_y], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p1y1 = np.array([[alpha, 0, p_x], [0,alpha, p_y+y1_err], [0,0,1]]) @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]
    p2y2 = np.array([[alpha, 0, p_x], [0,alpha, p_y+y2_err], [0,0,1]]) @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2cx = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0] + xc_err], [centre2[1]], [centre2[2]]])]
    p2cy = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0]], [centre2[1]+yc_err], [centre2[2]]])]
    p2cz = K @ R2 @ np.c_[np.eye(3), -np.array([[centre2[0]], [centre2[1]], [centre2[2]+zc_err]])]
    p2_yaw = K @ RotationMatrix(yaw+alpha_err, pitch, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_pitch = K @ RotationMatrix(yaw, pitch+beta_err, roll) @ np.c_[np.eye(3), -centre2.reshape(3,1)]
    p2_roll = K @ RotationMatrix(yaw, pitch, roll+gamma_err) @ np.c_[np.eye(3), -centre2.reshape(3,1)]

    # relative distance (mean)
    r = np.sum(np.sqrt((triangulate(imgpoint1_cam1, imgpoint1_cam2, p1, p2) - triangulate(imgpoint2_cam1, imgpoint2_cam2, p1, p2))**2))

    alpha_r_f1 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1f, p2) - r
    alpha_r_f2 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2f) - r
    alpha_r_x0_1 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1x1, p2) - r
    alpha_r_y0_1 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1y1, p2) - r
    alpha_r_x0_2 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2x2) - r
    alpha_r_y0_2 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2y2) - r
    alpha_r_yaw = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2_yaw) - r
    alpha_r_pitch = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2_pitch) - r
    alpha_r_roll = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2_roll) - r
    alpha_r_xc = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2cx) - r
    alpha_r_yc = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2cy) - r
    alpha_r_zc = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, imgpoint2_cam2, p1, p2cz) - r

    # vary image measurements independently in both cameras...
    alpha_r_u11 = relative_distance([imgpoint1_cam1[0]+imgcoordx_err, imgpoint1_cam1[1]], imgpoint1_cam2, 
                                    imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r # image of point 1 x-coord in camera 1

    alpha_r_v11 = relative_distance([imgpoint1_cam1[0], imgpoint1_cam1[1]+imgcoordy_err], imgpoint1_cam2, 
                                    imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r # image of point 1 y-coord in camera 1                      
    alpha_r_u12 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, [imgpoint2_cam1[0]+imgcoordx_err, 
                                   imgpoint2_cam1[1]], imgpoint2_cam2, p1, p2) - r  # image of point 2 x-coord in camera 1

    alpha_r_v12 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, [imgpoint2_cam1[0], 
                                   imgpoint2_cam1[1]+imgcoordy_err], imgpoint2_cam2, p1, p2) - r # image of point 2 y-coord in camera 1
                                   
    alpha_r_u21 = relative_distance(imgpoint1_cam1, [imgpoint1_cam2[0]+imgcoordx_err, imgpoint1_cam2[1]], 
                                    imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r # image of point 1 x-coord in camera 2

    alpha_r_v21 = relative_distance(imgpoint1_cam1, [imgpoint1_cam2[0], imgpoint1_cam2[1]+imgcoordy_err], 
                                    imgpoint2_cam1, imgpoint2_cam2, p1, p2) - r # image of point 1 y-coord in camera 2

    alpha_r_u22 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, [imgpoint2_cam2[0] + imgcoordx_err, 
                                    imgpoint2_cam2[1]], p1, p2) - r # image of point 2 x-coord in camera 2

    alpha_r_v22 = relative_distance(imgpoint1_cam1, imgpoint1_cam2, imgpoint2_cam1, [imgpoint2_cam2[0], 
                                    imgpoint2_cam2[1]+imgcoordy_err], p1, p2) - r # image of point 2 y-coord in camera 2

    total_error = np.sqrt(alpha_r_f1**2 + alpha_r_f2**2 + alpha_r_x0_1**2 + alpha_r_y0_1**2 + alpha_r_x0_2**2 + 
                          alpha_r_y0_2**2 + alpha_r_yaw**2 + alpha_r_pitch**2 + alpha_r_roll**2 + alpha_r_xc**2 + 
                          alpha_r_yc**2 + alpha_r_zc**2 + alpha_r_u11**2 + alpha_r_v11**2 + alpha_r_u12**2 + 
                          alpha_r_v12**2 + alpha_r_u21**2 + alpha_r_v21**2 + alpha_r_u22**2 + alpha_r_v22**2)

    # print("Error in relative distance due to f1 is ", alpha_r_f1)
    # print("Error in relative distance due to f2 is ", alpha_r_f2)
    # print("Error in relative distance due to x0_1 is ", alpha_r_x0_1)
    # print("Error in relative distance due to y0_1 is ", alpha_r_y0_1)
    # print("Error in relative distance due to x0_2 is ", alpha_r_x0_2)
    # print("Error in relative distance due to y0_2 is ", alpha_r_y0_2)
    # print("Error in relative distance due to yaw is ", alpha_r_yaw)
    # print("Error in relative distance due to pitch is ", alpha_r_pitch)
    # print("Error in relative distance due to roll is ", alpha_r_roll)
    # print("Error in relative distance due to camera 2 x coord is ", alpha_r_xc)
    # print("Error in relative distance due to camera 2 y coord is ", alpha_r_yc)
    # print("Error in relative distance due to camera 2 z coord is ", alpha_r_zc)
    # print("\nThe following are the errors for the measurement of of uij or vij for x or y coordinate in camera i of point j")
    # print("Error in relative distance due to u11 is ", alpha_r_u11)
    # print("Error in relative distance due to v11 is ", alpha_r_v11)
    # print("Error in relative distance due to u12 is ", alpha_r_u12)
    # print("Error in relative distance due to v12 is ", alpha_r_v12)
    # print("Error in relative distance due to u21 is ", alpha_r_u21)
    # print("Error in relative distance due to v21 is ", alpha_r_v21)
    # print("Error in relative distance due to u22 is ", alpha_r_u22)
    # print("Error in relative distance due to v22 is ", alpha_r_v22)

    # print("Total error in the relative distance between points is ", total_error, "m")

    errors = nnd_err(alpha, f_err, p_x, x_error, p_y, y_error, centre2[0], xc_err, 
                    centre2[1], yc_err, centre2[2], zc_err, yaw, alpha_err, pitch, beta_err, 
                    roll, gamma_err, imgcoordx_err, imgcoordy_err, np.array([25/2, 0, 100]), 1000,
                    radius=.5)

    
    print("mean error over many orientations is ", np.mean(errors), "m\n")

    point = np.array([25/2, 0, 100, 1])

    u1, v1 = homogeneous2Inhomogeneous(p1 @ point)
    u2, v2 = homogeneous2Inhomogeneous(p2 @ point)

    mean, error = functional_error(alpha, f_err, p_x, x_error, p_y, y_error, centre2[0], xc_err, 
                    centre2[1], yc_err, centre2[2], zc_err, yaw, alpha_err, pitch, beta_err, 
                    roll, gamma_err, u1, u2, imgcoordx_err, v1, v2, imgcoordy_err)
    print("Absolute distance errors:")
    print(mean, error)
    print(np.sqrt(np.sum(error**2)))

    return None

def projective_transform():

    centre1 = np.array([0,0,0])   # Camera 1 centre (origin)
    centre2 = np.array([25,0,0])  # Camera 2 centre
    R2 = RotationMatrix(yaw=-10, pitch=0, roll=0)   # Camera 2 orientation (compared to camera 1)
    R1 = np.eye(3)                                # Camera 1 orientation (none! )
    f = 50e-3                                       # Focal Length (m)
    m = 6000/23.5e-3                                # Pixels per unit length
    alpha = f*m                                     # Focal length in pixels   
    p_x, p_y = 3000, 2000                           # Centre pixels. (Prinicpal point)
    f_error = 1e-3*m                                # Focal length uncertainty (pixels)
    x_error = 5                                     # Uncertainty in position of principal point
    y_error = 5                                     # Uncertainty in position of principal point

    K = np.array([[alpha, 0, p_x], [0, alpha, p_y], [0,0,1]])   # Calibration matrix (same for both cameras...)

    p1 = K @ R1 @ np.c_[np.eye(3), -centre1.reshape(3,1)]       # Camera matrix 1
    p2 = K @ R2 @ np.c_[np.eye(3), -centre2.reshape(3,1)]       # Camera matrix 2


    points3d = np.array([[0, 0, 0, 1], # bottom lower left
                         [0, 0, 5, 1], # top lower left
                         [5, 0, 5, 1], # top lower right
                         [5, 0, 0, 1], # bottom lower right
                         [0, 5, 0, 1], # bottom upper left
                         [0, 5, 5, 1], # top upper left
                         [5, 5, 5, 1], # top upper right
                         [5, 5, 0, 1], # bottom upper right
                         [2.5, 0, 8, 1], # closest roof
                         [2.5, 2, 8, 1],
                         [2.5, 4, 8, 1], 
                         [2.5, 5, 8, 1]]) # furthest roof

    bll, tll, tlr, blr, bul, tul, tur, bur, cr, _, _, fr  = points3d[:]

    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2])

    ax.plot((bll[0], tll[0]), (bll[1], tll[1]), (bll[2], tll[2]), "k-")
    ax.plot((bll[0], blr[0]), (bll[1], blr[1]), (bll[2], blr[2]), "k-")
    ax.plot((tll[0], tlr[0]), (tll[1], tlr[1]), (tll[2], tlr[2]), "k-")
    ax.plot((blr[0], tlr[0]), (blr[1], tlr[1]), (blr[2], tlr[2]), "k-")

    ax.plot((bul[0], tul[0]), (bul[1], tul[1]), (bul[2], tul[2]), "k-")
    ax.plot((bul[0], bur[0]), (bul[1], bur[1]), (bul[2], bur[2]), "k-")
    ax.plot((tul[0], tur[0]), (tul[1], tur[1]), (tul[2], tur[2]), "k-")
    ax.plot((bur[0], tur[0]), (bur[1], tur[1]), (bur[2], tur[2]), "k-")

    ax.plot((cr[0], fr[0]), (cr[1], fr[1]), (cr[2], fr[2]), "k-")
    ax.plot((tll[0], cr[0]), (tll[1], cr[1]), (tll[2], cr[2]), "k-")
    ax.plot((tlr[0], cr[0]), (tlr[1], cr[1]), (tlr[2], cr[2]), "k-")
    ax.plot((tul[0], fr[0]), (tul[1], fr[1]), (tul[2], fr[2]), "k-")
    ax.plot((tur[0], fr[0]), (tur[1], fr[1]), (tur[2], fr[2]), "k-")

    ax.plot((bll[0], bul[0]), (bll[1], bul[1]), (bll[2], bul[2]), "k-")
    ax.plot((bur[0], bul[0]), (bur[1], bul[1]), (bur[2], bul[2]), "k-")
    ax.plot((blr[0], bur[0]), (blr[1], bur[1]), (blr[2], bur[2]), "k-")
    ax.plot((bll[0], blr[0]), (bll[1], blr[1]), (bll[2], blr[2]), "k-")
    ax.plot((tll[0], tul[0]), (tll[1], tul[1]), (tll[2], tul[2]), "k-")
    ax.plot((tur[0], tul[0]), (tur[1], tul[1]), (tur[2], tul[2]), "k-")
    ax.plot((tlr[0], tur[0]), (tlr[1], tur[1]), (tlr[2], tur[2]), "k-")
    ax.plot((tll[0], tlr[0]), (tll[1], tlr[1]), (tll[2], tlr[2]), "k-")

    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')

    # create animated gif of reconstruction...
    angles = np.linspace(0, 360, 26)[:-1]  # A list of 25 angles between 0 and 360
    # create an animated .gif
    for ii in range(0,360,3):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("house/movie{}.png".format(ii))

    # plt.show()

    counter = 0
    while counter < 5:
        try:
            H = np.random.uniform(0, 1, size=(4,4))
            invH = np.linalg.inv(H)
        except np.linalg.linalg.LinAlgError:
            continue

        newpoints = points3d.copy()
        for i in range(len(newpoints)):
            newpoints[i] = H @ newpoints[i]

        bll, tll, tlr, blr, bul, tul, tur, bur, cr, _, _, fr  = newpoints[:]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(newpoints[:,0], newpoints[:,1], newpoints[:,2])

        ax.plot((bll[0], tll[0]), (bll[1], tll[1]), (bll[2], tll[2]), "k-")
        ax.plot((bll[0], blr[0]), (bll[1], blr[1]), (bll[2], blr[2]), "k-")
        ax.plot((tll[0], tlr[0]), (tll[1], tlr[1]), (tll[2], tlr[2]), "k-")
        ax.plot((blr[0], tlr[0]), (blr[1], tlr[1]), (blr[2], tlr[2]), "k-")

        ax.plot((bul[0], tul[0]), (bul[1], tul[1]), (bul[2], tul[2]), "k-")
        ax.plot((bul[0], bur[0]), (bul[1], bur[1]), (bul[2], bur[2]), "k-")
        ax.plot((tul[0], tur[0]), (tul[1], tur[1]), (tul[2], tur[2]), "k-")
        ax.plot((bur[0], tur[0]), (bur[1], tur[1]), (bur[2], tur[2]), "k-")

        ax.plot((cr[0], fr[0]), (cr[1], fr[1]), (cr[2], fr[2]), "k-")
        ax.plot((tll[0], cr[0]), (tll[1], cr[1]), (tll[2], cr[2]), "k-")
        ax.plot((tlr[0], cr[0]), (tlr[1], cr[1]), (tlr[2], cr[2]), "k-")
        ax.plot((tul[0], fr[0]), (tul[1], fr[1]), (tul[2], fr[2]), "k-")
        ax.plot((tur[0], fr[0]), (tur[1], fr[1]), (tur[2], fr[2]), "k-")

        ax.plot((bll[0], bul[0]), (bll[1], bul[1]), (bll[2], bul[2]), "k-")
        ax.plot((bur[0], bul[0]), (bur[1], bul[1]), (bur[2], bul[2]), "k-")
        ax.plot((blr[0], bur[0]), (blr[1], bur[1]), (blr[2], bur[2]), "k-")
        ax.plot((bll[0], blr[0]), (bll[1], blr[1]), (bll[2], blr[2]), "k-")
        ax.plot((tll[0], tul[0]), (tll[1], tul[1]), (tll[2], tul[2]), "k-")
        ax.plot((tur[0], tul[0]), (tur[1], tul[1]), (tur[2], tul[2]), "k-")
        ax.plot((tlr[0], tur[0]), (tlr[1], tur[1]), (tlr[2], tur[2]), "k-")
        ax.plot((tll[0], tlr[0]), (tll[1], tlr[1]), (tll[2], tlr[2]), "k-")

        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zlabel('z')

        folder = '{}'.format(counter)
        if not os.path.exists('folder'):
            os.mkdir('{}'.format(folder))

        # create animated gif of reconstruction...
        angles = np.linspace(0, 360, 26)[:-1]  # A list of 25 angles between 0 and 360
        # create an animated .gif
        for ii in range(0,360,3):
            ax.view_init(elev=10., azim=ii)
            plt.savefig("{}/movie{}.png".format(folder,ii))

        plt.savefig('{}/{}.png'.format(folder,counter))

        # plt.show(fig)

        counter += 1


if __name__ == "__main__":
    # main()
    # test()
    # test2()
    # final()
    # fov()
    rel_dist()
    # projective_transform()   # demoing how a projective transform looks