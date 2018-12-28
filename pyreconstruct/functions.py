from __future__ import division, print_function

import random

import numpy as np
import scipy as sp

random.seed(10)  # this should remain constant for testing, remove for true random distribution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    u, d, vt = np.linalg.svd(P)
    assert np.allclose(u @ np.diag(d) @ vt, P), "SVD of P has not worked correctly..."
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

    assert np.allclose(d[0], d[1]), "The first two elements of the diagonal are not equal"
    assert d[-1] == 0, "The final diagonal element is not zero..."

    if not d[0] == 1:
        scale = 1 / d[0]  # force the diagonal elements to be (1,1,0)
        D = np.diag(scale * d)
        u = 1 / scale * u

    else:
        D = np.diag(d)
        pass

    assert np.allclose(u @ D @ vt, E), "SVD decomposition has not worked!"

    return u, D, vt


def depth(P, X):
    """Given a 3D point X = (x, y, z) and P = [M|p_4] determine the depth
    in front of the camera.
    
    Return Depth of point X."""
    if not len(X) == 4:
        X = np.r_[X, 1]  # turn into homgeneous coordinate...

    T = X[-1]
    M = P[:3, :3]  # 3x3 left hand submatrix
    p4 = P[:, -1]  # last column of p

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

    u, _, vt = decomposeEssentialMatrix(E)
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
    """Randomly select the coordinates of 5 (default) 3D points to act as ground truths to recover the 
    metric reconstruction of the projection..."""
    index = []

    while len(index) < 5:
        i = random.randint(0, len(arr_of_points) - 1)
        if not i in index:
            index.append(i)
            continue
        continue

    points = np.zeros((5, 3))

    print(index)
    for i in range(len(index)):
        points[i] = arr_of_points[index[i]]

    return points


class Camera:
    """Define Camera centre, focal lenght, sensor dimensions and pixel dimensions..."""

    def __init__(self, cameracentre, focal_length, sensor_width, sensor_height, pixel_size):
        self.centre = cameracentre
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.pixel_size = pixel_size


class Sim:
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
            print((x1, y1))
            print((x2, y1))

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
    point3D = np.array([[0, 0, 50]])
    camera2centre = np.array([0, 0, 0])

    Cam1 = Camera(cameracentre, focal_length, sensor_width, 
                  sensor_height, pixel_size)

    Cam2 = Camera(camera2centre, focal_length, sensor_width, 
                  sensor_height, pixel_size)

    sim = Sim(200, Cam1, Cam2, yaw=0, pitch=45, roll=0)

    # x1s, y1s, x2s, y2s, seenpoints = sim.synchImages()

    # sim.drawImages(x1s, y1s, x2s, y2s)

    sim.scene3D()

    # sim.seenpoints3D(seenpoints)
    # sim.testPoint(np.array([0,0,1]))
    # sim.testPoint(np.array([0,0,2]))
    # sim.testPoint(np.array([0,0,5e5]))

    # points = sim.returnPoints()

    # print("3D point is at: ", points)

    plt.show()


if __name__ == "__main__":
    main()
