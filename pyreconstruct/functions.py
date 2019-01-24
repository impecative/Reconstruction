from __future__ import division, print_function
import numpy as np
import scipy as sp
import scipy.linalg
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
random.seed(10)  # this should remain constant for testing, remove for true random distribution

# __author__ = "Alex Elliott"
# __version__ = "0.02"

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

def RotationMatrix(yaw, pitch, roll):
    '''Form a rotation matrix to carry out specified yaw, pitch, roll rotation *in degrees*.'''
    yaw, pitch, roll = np.radians(yaw), np.radians(pitch), np.radians(roll)

    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    R = np.linalg.multi_dot([R_z, R_y, R_x])

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
    ax.plot([prTL1[0], prTR1[0]], [prTL1[1], prTR1[1]], [prTL1[2], prTR1[2]], c="r", alpha=0.5, label="CCD Boundaries")
    ax.plot([prTR1[0], prBL1[0]], [prTL1[1], prBL1[1]], [prTR1[2], prBL1[2]], c="r", alpha=0.5)
    ax.plot([prBL1[0], prBR1[0]], [prBL1[1], prBR1[1]], [prBL1[2], prBR1[2]], c="r", alpha=0.5)
    ax.plot([prTL1[0], prBR1[0]], [prTL1[1], prBR1[1]], [prTL1[2], prBR1[2]], c="r", alpha=0.5)
    ax.plot([prTL1[0], cameracentre1[0]], [prTL1[1], cameracentre1[1]], [prTL1[2], cameracentre1[2]], c="y", alpha=0.5)
    ax.plot([prTR1[0], cameracentre1[0]], [prTR1[1], cameracentre1[1]], [prTR1[2], cameracentre1[2]], c="y", alpha=0.5)
    ax.plot([prBL1[0], cameracentre1[0]], [prBL1[1], cameracentre1[1]], [prBL1[2], cameracentre1[2]], c="y", alpha=0.5)
    ax.plot([prBR1[0], cameracentre1[0]], [prBR1[1], cameracentre1[1]], [prBR1[2], cameracentre1[2]], c="y", alpha=0.5)

    ax.plot([prTL2[0], prTR2[0]], [prTL2[1], prTR2[1]], [prTL2[2], prTR2[2]], c="r", alpha=0.5)
    ax.plot([prTR2[0], prBL2[0]], [prTL2[1], prBL2[1]], [prTR2[2], prBL2[2]], c="r", alpha=0.5)
    ax.plot([prBL2[0], prBR2[0]], [prBL2[1], prBR2[1]], [prBL2[2], prBR2[2]], c="r", alpha=0.5)
    ax.plot([prTL2[0], prBR2[0]], [prTL2[1], prBR2[1]], [prTL2[2], prBR2[2]], c="r", alpha=0.5)
    ax.plot([prTL2[0], cameracentre2[0]], [prTL2[1], cameracentre2[1]], [prTL2[2], cameracentre2[2]], c="y", alpha=0.5)
    ax.plot([prBL2[0], cameracentre2[0]], [prBL2[1], cameracentre2[1]], [prBL2[2], cameracentre2[2]], c="y", alpha=0.5)
    ax.plot([prTR2[0], cameracentre2[0]], [prTR2[1], cameracentre2[1]], [prTR2[2], cameracentre2[2]], c="y", alpha=0.5)
    ax.plot([prBR2[0], cameracentre2[0]], [prBR2[1], cameracentre2[1]], [prBR2[2], cameracentre2[2]], c="y", alpha=0.5)

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

    return T1, T2

def translateFundamentalMatrix(F, T1, T2):
    """Replace matrix F with transpose(inv(T2)) F inv(T1)."""
    T2inv = np.linalg.inv(T2)
    T1inv = np.linalg.inv(T1)
    return np.linalg.multi_dot([T2inv.T, F, T1inv])

def findEpipoles(F):
    """
    Find the right and left epipoles e1 and e2 such that 
    e'.T F = 0 and F e = 0. 
    
    Return normalsed epipoles...
    """

    # e1 is the right null-space of F, e2.T is left null space, so use SVD. 
    e1 = right_null_space(F)
    e2 = left_null_space(F).T

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

def getRotationMatrices(e1, e2):
    """Given two epipoles e1 = (e11, e12, e13)^T and 
    e2 = (e21, e22, e23)^T, return rotation matrices R
    and R' such that Re1 = (1, 0, e13)^t and Re2 = (1,0,e23)^T)."""
    e11, e12, _ = e1.ravel()
    e21, e22, _ = e2.ravel()

    R1 = np.array([[e11, e12, 0], [-e12, e11, 0], [0, 0, 1]])
    R2 = np.array([[e21, e22, 0], [-e22, e21, 0], [0, 0, 1]])

    return R1, R2

def rotateFundamentalMatrix(F, R1, R2):
    """Replace matrix F with (R2 F R1.T)"""
    return np.linalg.multi_dot([R2, F, R1.T])

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
        val = np.polyval(coefs, t.real)
        if np.isclose(val, 0):
            counter += 1

    assert counter > 0, "No roots were found..."

    return roots

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

    newx1 = np.linalg.multi_dot([np.linalg.inv(T1), R1.T, x1])
    newx2 = np.linalg.multi_dot([np.linalg.inv(T2), R2.T, x2])

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
    return right_null_space(A).reshape(4,1)

def homogeneous2Inhomogeneous(X):
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
    projection matrices P1 and P2 respectively...
    
    Return an INHOMOGENEOUS 3-vector coordinate of the point \
    corresponding to the image in camera 1 with camera matrix P1 and \
    the image in camera 2 with camera matrix P2. '''
    # first form the matrix A
    A = formA(imgcoord1, imgcoord2, P1, P2)
    
    # find point X in homogeneous coordinates
    X = find3DPoint(A)

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
    print("Diagonal elements of newF are : ", d)

    # print("F = ", newF)

    print("F has rank({})".format(np.linalg.matrix_rank(newF)))

    assert np.linalg.matrix_rank(newF) == 2, "Computed F has rank({}), the correct F should have rank(2)...".format(np.linalg.matrix_rank(newF))
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

    e2_skew = skew(e2)
    # print("The skew matrix e2 is ", e2_skew)

    leftMatrix = e2_skew @ F 
    P2 = np.c_[leftMatrix, e2]

    return P1, P2

def cameraMatrices(img1_points, img2_points):
    """
    Determine the camera matrices p1, p2 and the fundemental matrix 
    relating image correspondences in image 1 and image 2.

    Inputs:
    img1_points : np.array of image coordinates in first camera.
    img2_points : np.array of image coordinate in second camera.
    NOTE : The points must be corresponding, and in the same order, in 
           img1_points and img2_points. 

    Out: P1, P2, F
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
        T1, T2 = getTransformationMatrices(img1coords[i], img2coords[i])
        newF = translateFundamentalMatrix(F, T1, T2)
        e1, e2 = findEpipoles(newF)
        R1, R2 = getRotationMatrices(e1, e2)
        newF = rotateFundamentalMatrix(newF, R1, R2)
        formPolynomial(e1, e2, newF)
        a,b,c,d,f,g = formPolynomial(e1, e2, newF)
        roots = solvePolynomial(a,b,c,d,f,g)
        tmin = evaluateCostFunction(roots, a,b,c,d,f,g)
        x1, x2 = findModelPoints(tmin,a,b,c,d,f,g)  # These are the corrected point correspondences
        newx1, newx2 = findOriginalCoordinates(R1, R2, T1, T2, x1, x2) # transform back to original coordinates.

        # newx1 and newx2 are the optimal point correspondences! 
        # Now use homogeneous triangulation method to compute 3D point.
        X = triangulate(newx1, newx2, p1, p2)
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

class Camera:
    """
    Define Camera centre, focal lenght, sensor dimensions and pixel \
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
    def __init__(self, no_of_points, Cam1, Cam2, yaw, pitch, roll):
        self.tvec = Cam2.centre - Cam1.centre
        self.R = RotationMatrix(yaw, pitch, roll)

        points = np.zeros((no_of_points, 3))
        for i in range(points.shape[0]):
            points[i][0] = random.uniform(-10, 10)  # xs
            points[i][1] = random.uniform(-10, 10)  # ys
            points[i][2] = random.uniform(45, 55)  # zs

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
    camera2centre = np.array([0, 0, 0])

    Cam1 = Camera(cameracentre, focal_length, sensor_width, 
                  sensor_height, pixel_size)

    Cam2 = Camera(camera2centre, focal_length, sensor_width, 
                  sensor_height, pixel_size)

    sim = Sim(200, Cam1, Cam2, yaw=0, pitch=-12, roll=0)

    x1s, y1s, x2s, y2s, seenpoints = sim.synchImages()

    sim.drawImages(x1s, y1s, x2s, y2s)

    sim.scene3D()

    # sim.seenpoints3D(seenpoints)
    # sim.testPoint(np.array([0,0,1]))
    # sim.testPoint(np.array([0,0,2]))
    # sim.testPoint(np.array([0,0,5e5]))

    # points = sim.returnPoints()

    # print("3D point is at: ", points)

    points_triangulated = sim.reconstruction()
    # print(points_triangulated[:10])
    # print(seenpoints[:10])



    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for point in points_triangulated:
        x, y, z = homogeneous2Inhomogeneous(point)

        ax.scatter(x, y, z)

    for point in seenpoints:
        x, y, z = point
        ax.scatter(x,y,z, c="k", alpha=0.2)


    plt.show()


if __name__ == "__main__":
    main()
