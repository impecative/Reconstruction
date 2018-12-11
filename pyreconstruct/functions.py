from __future__ import division, print_function
import numpy as np
import scipy as sp
import random
random.seed(10)  # this should remain constant for testing, remove for true random distribution

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
    
def linepoint2point(A,B,t):
    '''Find the equation of the line from point A -> B. \n
    Returns a point on the line P satisfies P = A + (B-A)*t for some t.'''
    return A + (B-A)*t

def findPlane(A, B, C, *args):
    '''Given three points A, B, C find the equation of the plane they all lie upon.
    Can input more points to verify if they are all coplanar. If testing the sensor 
    plane, this is recommended.
    \nReturns a,b,c of the normal=(a,b,c) and the scalar component np.dot(normal, A).'''
    normal = np.cross((B-A), (C-A))
    d = np.dot(normal, A)
    a,b,c = normal

    if args:
        for coord in args:
            assert np.allclose(a*coord[0] + b*coord[1] + c*coord[2] -d, 0), "additional input point is not coplanar..."

    return a,b,c,d

def point2plane2point(a, b, c, d, point3D, cameracentre):
    '''Where does the line joining the 3D point and camera centre intersect the plane? 
    Return the x,y,z position of this intercept. '''
    n = np.array([a,b,c])
    D = point3D
    E = cameracentre

    t = (d - np.linalg.multi_dot([n, D]))/np.linalg.multi_dot([n, (E-D)])

    # point of intersection is 
    x,y, z = D + (E-D)*t

    return x,y,z

def RotationMatrix(yaw, pitch, roll):
    '''Form a rotation matrix to carry out specified yaw, pitch, roll rotation *in degrees*.'''
    yaw, pitch, roll = np.radians(yaw), np.radians(pitch), np.radians(roll)

    R_x = np.array([[1,0,0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],[0,1,0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw),0], [0,0,1]])

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

def pointInCamera1Image(point3D, sensor_width, sensor_height, focal_length, pixel_size, cameracentre=np.array([0,0,0])):
    '''Can camera 1 (at the origin of the coordinate system) see the 3D point in space? \n
    Return True/False and the x,y pixel coordinates of the image.'''
    w, h =sensor_width, sensor_height
    f = focal_length

    TL = np.array([-w/2, h/2, f])   # top left of sensor
    TR = np.array([w/2, h/2, f])    # top right of sensor
    BR = np.array([w/2, -h/2, f])   # bottom right of sensor
    BL = np.array([-w/2, -h/2, f])  # bottom left of sensor

    # define limits on the sensor    (A point exceeding these dimensions cannot be seen)
    xmin, ymin, _ = BL
    xmax, ymax, _ = TR

    # define the plane of the sensor and the line linking the 3D point and the camera centre.
    # Where they intersect is the image coordinates of the image. -> then we can check whether it is in frame...

    # pick 3 corners of the four to define plane.
    a,b,c,d= findPlane(TL, TR, BR, BL)
    intersection = point2plane2point(a,b,c,d,point3D, cameracentre)

    # can the point be seen? 
    x,y,z = intersection

    # print(z, f)
    assert np.allclose(z,f), "Intersection of image plane hasn't worked properly... point not at z=f"

    seen = False
    if (x >= xmin and x <= xmax) and (y >= ymin and y <= ymax):
        seen = True
    
    # check if the point is in front of the CCD plane: 
    if point3D[2] < f:    # if the point's z-position is less than the z-position of the CCD plane
        seen = False      # then the point cannot be physically imaged by the camera.
    
    # return the pixel coordinates of the pixel... 
    # NOTE: we are treating the origin as the CENTRE of the image, not the bottom left corner. 
    x = x/pixel_size
    y = y/pixel_size
    
    return seen, x, y

def pointInCamera2Image(point3D, sensor_width, sensor_height, focal_length, pixel_size, cameracentre, tvec, rotation_matrix):
    '''Can camera 2 at a given translation vector and rotation from camera 1 see the 3D point? \n
    Return True/False and x, y coordinate of point in the image.'''
    w, h =sensor_width, sensor_height
    f = focal_length

    # if the camera centre was at the origin and looking straight down the +ve z-direction
    TL = np.array([-w/2, h/2, f])   # top left of sensor
    TR = np.array([w/2, h/2, f])    # top right of sensor
    BR = np.array([w/2, -h/2, f])   # bottom right of sensor
    BL = np.array([-w/2, -h/2, f])  # bottom left of sensor
    sensor_centre = np.mean([TL, TR, BR, BL], axis=0)

    # define limits on the sensor
    xmin, ymin, _ = BL
    xmax, ymax, _ = TR

    # now apply transformation so everything is where it 'should be' in 3-space
    R = rotation_matrix
    newcameracentre = cameracentre   # don't need to transform this! 
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
    a,b,c,d = findPlane(TL2, TR2, BR2, BL2)
    intersection = point2plane2point(a, b, c, d, point3D, cameracentre)

    # this intersection is in the transformed frame. Untransform and we can see if it exceeds the 
    # sensor dimensions, and therefore whether the point can be seen on the CCD.


    x,y,z = untransformpoint(intersection, R, tvec)
    assert np.allclose(f,z), "Intersection of image plane hasn't worked properly... point not at z=f"

    seen = False
    if (x >= xmin and x <= xmax) and (y >= ymin and y <= ymax) and front:
        seen = True
    
    # turn the coordinates into pixel coordinates
    x = x/pixel_size
    y = y/pixel_size
    
    return seen, x, y

def fixImgCoords(imgx, imgy, sensor_width, sensor_height):
    '''Return the image coordinates as a np.array with the origin at the bottom left corner of the image.'''
    w, h = sensor_width, sensor_height
    origin = np.array([-w/2, -h/2])
    imgcoordinate = np.array([imgx, imgy])

    # relative to the origin, the point is at position:
    return imgcoordinate - origin

class Camera:
    """Define Camera centre, focal lenght, sensor dimensions and pixel dimensions..."""
    def __init__(self, cameracentre, focal_length, sensor_width, sensor_height, pixel_size):
        self.centre        = cameracentre
        self.focal_length  = focal_length
        self.sensor_width  = sensor_width
        self.sensor_height = sensor_height
        self.pixel_size    = pixel_size
    
class Sim:
    def __init__(self, no_of_points,  Cam1, Cam2, yaw, pitch, roll):
        self.tvec       = Cam2.centre - Cam1.centre
        self.R          = RotationMatrix(yaw, pitch, roll)
    
        points = np.zeros((no_of_points, 3))
        for i in range(points.shape[0]):
            points[i][0] = random.uniform(-10, 10)    # xs
            points[i][1] = random.uniform(-10, 10)    # ys
            points[i][2] = random.uniform(5, 100)     # zs

        self.points3D   = points
        self.w1         = Cam1.sensor_width 
        self.h1         = Cam1.sensor_height
        self.f1         = Cam1.focal_length
        self.p1         = Cam1.pixel_size
        self.w2         = Cam2.sensor_width 
        self.h2         = Cam2.sensor_height
        self.f2         = Cam2.focal_length
        self.p2         = Cam2.pixel_size
        self.camera1centre = Cam1.centre
        self.camer2centre = Cam2.centre
        self.R = RotationMatrix(yaw, pitch, roll)
        self.tvec = Cam2.centre - Cam1.centre

    def synchImages(self):
        """Check if 3D point is visible to BOTH cameras. 
        \nIf so, then return it's pixel coordinates in each image... """
        x1s, y1s = [], []
        x2s, y2s = [], []
        for point in self.points3D:
            seen1, x1, y1 = pointInCamera1Image(point, self.w1, self.h1, self.f1, self.p1, self.camera1centre)
            seen2, x2, y2 = pointInCamera2Image(point, self.w2, self.h2, self.f2, self.p2, self.camer2centre, self.tvec, self.R)   

            if seen1 and seen2:
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)
            
        print("{} out of {} points can be seen in both images".format(len(x1s), len(self.points3D)))
        
        # fix the coordinates, so image origin is bottom left corner of CCD
        for i in range(len(x1s)):
            x1, y1 = fixImgCoords(x1s[i], y1s[i], self.w1/self.p1, self.h1/self.p1)
            x2, y2 = fixImgCoords(x2s[i], y2s[i], self.w2/self.p2, self.h2/self.p2)

            x1s[i] = x1
            y1s[i] = y1
            x2s[i] = x2
            y2s[i] = y2

        return x1s, y1s, x2s, y2s


    
cameracentre = np.array([0,0,0])   # let camera 1 lie at the origin of the coordinate system (m)
sensor_width, sensor_height = 23.5e-3, 15.6e-3     # sensor dimensions of first camera (m)
focal_length = 50e-3   # focal length of camera 1 (m)
pixel_size  = 3.9e-6   # linear dimension of a pixel (m)
point3D = np.array([[0,0,50]])

Cam1 = Camera(cameracentre, focal_length, sensor_width, sensor_height, pixel_size)

Cam2 = Camera(np.array([0,0,100]), focal_length, sensor_width, sensor_height, pixel_size)

sim = Sim(200, Cam1, Cam2, yaw=0, pitch=180, roll=0)

x1s, y1s, x2s, y2s = sim.synchImages()
