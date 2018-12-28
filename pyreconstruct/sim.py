from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import random
from n8p import *
from functions import *
from a_12_1 import *
import cv2

# to get arrows pointing in 3D # 
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)






def linepoint2point(A,B,t):
    '''Find the equation of the line from point A -> B. \n
    A point on the line P satisfies P = A + (B-A)*t for some t.'''
    return A + (B-A)*t

def findPlane(A, B, C, *args):
    '''Given three points A, B, C find the equation of the plane they all lie upon.
    Can input more points to verify if they are all coplanar. If testing the sensor 
    plane, this is recommended.'''
    normal = np.cross((B-A), (C-A))
    d = np.dot(normal, A)
    a,b,c = normal

    if args:
        for coord in args:
            assert np.allclose(a*coord[0] + b*coord[1] + c*coord[2] -d, 0), "additional input point is not coplanar..."

    return a,b,c,d, normal

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
    a,b,c,d, _ = findPlane(TL, TR, BR, BL)
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
    a,b,c,d, _ = findPlane(TL2, TR2, BR2, BL2)
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


def main():
    no_of_points = 200   # how many random points to generate? 

    # define the cameras:
    camera1centre = np.array([0,0,0])   # let camera 1 lie at the origin of the coordinate system (m)
    sensor1_width, sensor1_height = 23.5e-3, 15.6e-3     # sensor dimensions of first camera (m)
    focal_length1 = 50e-3   # focal length of camera 1 (m)
    pixel_size1  = 3.9e-6   # linear dimension of a pixel (m)
    sensor1centre = np.array([0,0,focal_length1])

    camera2centre = np.array([20, 0, 0])  # location of the second camera sensor
    sensor2_width, sensor2_height = 23.5e-3, 15.6e-3    # sensor dimensions of second camera (m) - not necessary to be equal to camera 1
    focal_length2 = 50e-3   # focal length of camera 2 (m)
    pixel_size2   = 3.9e-6  # linear dimension of a pixel (m)

    tvec = camera2centre - camera1centre   # translation vector between camera 1 and camera 2
    R = RotationMatrix(yaw=0, pitch=-12, roll=0)

    # calibration matrices
    K1 = np.array([[focal_length1/pixel_size1, 0, (sensor1_width/pixel_size1 /2)], 
                   [0, focal_length1/pixel_size1,  (sensor1_height/pixel_size1 /2)], [0,0,1]])
    K2 = np.array([[focal_length2/pixel_size2, 0, (sensor2_width/pixel_size2 /2)], 
                   [0, focal_length2/pixel_size2, (sensor2_height/pixel_size2 /2)], [0,0,1]])
    P1 = K1 @ np.c_[np.eye(3), np.array([0,0,0])]
    P2 = K2 @ np.c_[R, tvec]

    print(P1, P2)

    # print("rotation matrix R = ", R)
    sensor2centre = transformpoint(sensor1centre, R, tvec)

    # define the 3D point(s) to observe with the cameras...
    points = np.zeros((no_of_points, 3))
    random.seed(10)

    for i in range(points.shape[0]):
        points[i][0] = random.uniform(-10, 10)    # xs
        points[i][1] = random.uniform(-10, 10)    # ys
        points[i][2] = random.uniform(5, 100)     # zs

    # # for testing the image is the same in both cameras:
    # points = np.array([[0,0,50]])
    # print(points)


    # store the coordinates of the points visible on both images.
    img1xcoords = []
    img1ycoords = []
    img2xcoords = []
    img2ycoords = []

    counter = 0   # how many matched points are there? 

    # plot the points that can be seen by both sensors
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")

    for point in points:
        x,y,z = point
        # is it seen by camera 1?
        seen1, x1, y1 = pointInCamera1Image(point, sensor1_width, sensor1_height, focal_length1, pixel_size1, camera1centre)

        # is it seen by camera 2? 
        seen2, x2, y2 = pointInCamera2Image(point, sensor2_width, sensor2_height, focal_length2, pixel_size2, camera2centre, tvec, R)  # TBD

        # need to be seen by both cameras to be a matched point...

        # print("Seen by camera 1? ", seen1)
        # print("Seen by camera 2? ", seen2)

        # print(x1, y1)
        # print(x2, y2)

        if seen1 and seen2:  # seen by both cameras
            ax.scatter(x,y,z, c="g")
            
            ax.plot([camera1centre[0], x], [camera1centre[1], y], [camera1centre[2], z], c="g", alpha=0.3)
            ax.plot([camera2centre[0], x], [camera2centre[1], y], [camera2centre[2], z], c="m", alpha=0.3)

            img1xcoords.append(x1)
            img1ycoords.append(y1)
            img2xcoords.append(x2)
            img2ycoords.append(y2)
            counter += 1
    ax.scatter(camera1centre[0], camera1centre[1], camera1centre[2], c="m", label="Camera 1 Centre")
    ax.scatter(camera2centre[0], camera2centre[1], camera2centre[2], c="b", label="Camera 2 Centre")
    ax.set_title("The 3D Points That Are Observed by Both Cameras")
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")
    ax.legend()
    
    print("Both cameras can see {} out of the total {} points".format(counter, len(points)))

    w1, h1, w2, h2 = sensor1_width/pixel_size1, sensor1_height/pixel_size1, sensor2_width/pixel_size2, sensor2_height/pixel_size2

    # fix the coordinates so the images are at the origin
    for i in range(len(img1xcoords)):
        x1, y1 = fixImgCoords(img1xcoords[i], img1ycoords[i], w1, h1)
        x2, y2 = fixImgCoords(img2xcoords[i], img2ycoords[i], w2, h2)

        img1xcoords[i] = x1
        img1ycoords[i] = y1
        img2xcoords[i] = x2
        img2ycoords[i] = y2

    # plot the two camera images
    fig = plt.figure(2)
    ax1 = plt.subplot(121)
    plt.plot(img1xcoords, img1ycoords, "bx")
    # plot the sensor on
    plt.plot([0,w1], [0, 0], "r", alpha=0.3)
    plt.plot([w1,w1], [0, h1], "r", alpha=0.3)
    plt.plot([w1,0], [h1, h1], "r", alpha=0.3)
    plt.plot([0, 0], [h1, 0], "r", alpha=0.3)

    ax1.set_aspect("equal")
    plt.xlabel("x-direction (pixels)")
    plt.ylabel("y-direction (pixels)")
    plt.title("Camera 1 Image")
    
    ax2 = plt.subplot(122, sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.plot(img2xcoords, img2ycoords, "bx")
    
    # plot the sensor on
    plt.plot([0,w2], [0, 0], "r", alpha=0.3)
    plt.plot([w2,w2], [0, h2], "r", alpha=0.3)
    plt.plot([w2,0], [h2, h2], "r", alpha=0.3)
    plt.plot([0, 0], [h2, 0], "r", alpha=0.3)

    ax2.set_aspect("equal")
    plt.title("Camera 2 Image")
    plt.xlabel("x-direction (pixels)")

    plt.suptitle("Points as Imaged by Two Stereo-Cameras")
    # plt.show()

    # plot the origin 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs2, ys2, zs2 = points[:,0], points[:,1], points[:,2]
    ax.scatter(xs2, ys2, zs2, c="r")
    ax.scatter(camera1centre[0], camera1centre[1], camera1centre[2], c="m", label="Camera 1 Centre")
    ax.scatter(camera2centre[0], camera2centre[1], camera2centre[2], c="b", label="Camera 2 Centre")

    # plot the camera orientations...
    sensor1line = linepoint2point(camera1centre, sensor1centre, 800)
    sensor2line = linepoint2point(camera2centre, sensor2centre, 200)

    a = Arrow3D([camera1centre[0], sensor1line[0]], [camera1centre[1], sensor1line[1]], [camera1centre[2], sensor1line[2]], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="g")
    b = Arrow3D([camera2centre[0], sensor2line[0]], [camera2centre[1], sensor2line[1]], [camera2centre[2], sensor2line[2]], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="g")
    ax.add_artist(a)
    ax.add_artist(b)

    # ax.plot([sensor1line[0], camera1centre[0]], [sensor1line[1], camera1centre[1]], [sensor1line[2], camera1centre[2]], c="g")
    # ax.plot([sensor2line[0], camera2centre[0]], [sensor2line[1], camera2centre[1]], [sensor2line[2], camera2centre[2]], c="g")

    ax.set_title("3D Generated Points and Camera orientations")
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")
    ax.legend()
    # plt.show()

    ########################## 3D reconstruction #################################
    assert len(img1xcoords) >= 8, "Only have {} point matches, we need >=8".format(len(img1xcoords))

    img1coords = np.zeros((len(img1xcoords), 3))
    img2coords = np.zeros(img1coords.shape)
    img1coords[:,-1], img2coords[:,-1] = 1,1

    for i in range(len(img1coords)):
        img1coords[i][0] = img1xcoords[i]
        img1coords[i][1] = img1ycoords[i]
        img2coords[i][0] = img2xcoords[i]
        img2coords[i][1] = img2ycoords[i]

    # # from the reconstruction - seems to be a scale ambiguity
    # p1, p2 = cameraMatrices(img1coords, img2coords)
    # OpenCVF = cv2.findFundamentalMat(img1coords, img2coords)[0]
    # print(p1, p2)
    # print(findCameras(OpenCVF))

    # CVP1, CVP2 = findCameras(OpenCVF)
    # cvK1, cvR1, cvC1 = decomposeCameraMtx(CVP1)
    # cvK2, cvR2, cvC2 = decomposeCameraMtx(CVP2)

    # # print("Open CV finds the following properties of the second camera: ")
    # # print("Calibration matrix is: ")
    # # print(cvK2)
    # # print("Rotation matrix is: ")
    # # print(cvR2)

    # K1, R1, C1 = decomposeCameraMtx(p1)
    # K2, R2, C2 = decomposeCameraMtx(p2)

    # # print("Camera matrix 1 has calibration matrix, rotation matrix and centre: ")
    # # print(K1)
    # # print(R1)
    # # print(C1)
    # # print("\nCamera matrix 2 has calibration matrix, rotation matrix and centre: ")
    # # print(K2)
    # # print(R2)
    # # print(C2)


    # # Now use my own camera matrices...
    # K = np.array([[195e-9, 0,0], [0,195e-9, 0], [0,0,1]])
    # R1 = np.eye(3)
    # C1 = camera1centre
    # R2 = R
    # C2 = camera2centre
    
    # P1 = np.c_[K, np.zeros((3,1))]
    # lastcol2 = - R2 @ C2 
    # P2 = K @ np.c_[R2, lastcol2]

    points3D_measured = np.zeros((len(img1coords), 3))

    for i in range(len(img1coords)):
        x1, x2 = img1coords[i], img2coords[i]

        # now triangulate the points newx1, newx2 back to a 3D point X
        X = triangulate(x1, x2, P1, P2)

        # now store this 3D point
        points3D_measured[i] = X

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = points3D_measured[:,0], points3D_measured[:,1], points3D_measured[:,2]
    ax.scatter(xs, ys, zs, c="b")
    ax.set_xlabel("x ")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    points3D = estimate3DPoints(img1coords, img2coords)

    xs, ys, zs = points3D[:,0], points3D[:,1], points3D[:,2]
    
    # reconstructed positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c="g")
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")
    
    # plot the origin 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs2, ys2, zs2 = points[:,0], points[:,1], points[:,2]
    ax.scatter(xs2, ys2, zs2, c="r")
    ax.scatter(camera1centre[0], camera1centre[1], camera1centre[2], c="b")
    ax.scatter(camera2centre[0], camera2centre[1], camera2centre[2], c="b")

    # plot the camera orientations...
    sensor1line = linepoint2point(camera1centre, sensor1centre, 800)
    sensor2line = linepoint2point(camera2centre, sensor2centre, 200)

    a = Arrow3D([camera1centre[0], sensor1line[0]], [camera1centre[1], sensor1line[1]], [camera1centre[2], sensor1line[2]], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="g")
    b = Arrow3D([camera2centre[0], sensor2line[0]], [camera2centre[1], sensor2line[1]], [camera2centre[2], sensor2line[2]], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="g")
    ax.add_artist(a)
    ax.add_artist(b)

    # ax.plot([sensor1line[0], camera1centre[0]], [sensor1line[1], camera1centre[1]], [sensor1line[2], camera1centre[2]], c="g")
    # ax.plot([sensor2line[0], camera2centre[0]], [sensor2line[1], camera2centre[1]], [sensor2line[2], camera2centre[2]], c="g")

    ax.set_title("3D Generated Points and Cameras")
    ax.set_xlabel("x ")
    ax.set_ylabel("y ")
    ax.set_zlabel("z ")

    # plt.show()

main()
    
    









