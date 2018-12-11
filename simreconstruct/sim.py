from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random

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
    a,b,c,d = findPlane(TL, TR, BR, BL)
    intersection = point2plane2point(a,b,c,d,point3D, cameracentre)

    # can the point be seen? 
    x,y,z = intersection

    # print(z, f)
    assert np.allclose(z,f), "Intersection of image plane hasn't worked properly... point not at z=f"

    seen = False
    if (x >= xmin and x <= xmax) and (y >= ymin and y <= ymax):
        seen = True
    
    # return the pixel coordinates of the pixel... 
    # NOTE: we are treating the origin as the CENTRE of the image, not the bottom left corner. 
    x = x/pixel_size
    y = y/pixel_size
    
    return seen, x, y

def RotationMatrix(yaw, pitch, roll):
    '''Form a rotation matrix to carry out specified yaw, pitch, roll rotation *in degrees*.'''

    R_x = np.array([[1,0,0], [0, np.cos(np.deg2rad(roll)), -np.sin(np.deg2rad(roll))], [0, np.sin(np.deg2rad(roll)), np.cos(np.deg2rad(roll))]])
    R_y = np.array([[np.cos(np.deg2rad(pitch)), 0, np.sin(np.deg2rad(pitch))],[0,1,0], [-np.sin(np.deg2rad(pitch)), 0, np.cos(np.deg2rad(pitch))]])
    R_z = np.array([[np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw)), 0], [np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw)),0], [0,0,1]])

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

    # define limits on the sensor
    xmin, ymin, _ = BL
    xmax, ymax, _ = TR

    # now apply transformation so everything is where it 'should be' in 3-space
    R = rotation_matrix
    newcameracentre = transformpoint(cameracentre, R, tvec)
    TL2 = transformpoint(TL, R, tvec)
    TR2 = transformpoint(TR, R, tvec)
    BR2 = transformpoint(BR, R, tvec)
    BL2 = transformpoint(BL, R, tvec)


    # define the plane of these points
    # also find intersection of the line from 3Dpoint and the camera centre, and the plane
    a,b,c,d = findPlane(TL2, TR2, BR2, BL2)
    intersection = point2plane2point(a, b, c, d, point3D, cameracentre)

    # this intersection is in the transformed frame. Untransform and we can see if it exceeds the 
    # sensor dimensions, and therefore whether the point can be seen on the CCD.


    x,y,z = untransformpoint(intersection, R, tvec)
    assert np.allclose(f,z), "Intersection of image plane hasn't worked properly... point not at z=f"

    seen = False
    if (x >= xmin and x <= xmax) and (y >= ymin and y <= ymax):
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

    camera2centre = np.array([5, 0, 0])  # location of the second camera sensor
    sensor2_width, sensor2_height = 23.5e-3, 15.6e-3    # sensor dimensions of second camera (m) - not necessary to be equal to camera 1
    focal_length2 = 50e-3   # focal length of camera 2 (m)
    pixel_size2   = 3.9e-6  # linear dimension of a pixel (m)

    tvec = camera2centre - camera1centre   # translation vector between camera 1 and camera 2
    R = RotationMatrix(yaw=0, pitch=-20, roll=0)

    # define the 3D point(s) to observe with the cameras...
    points = np.zeros((no_of_points, 3))
    for i in range(points.shape[0]):
        points[i][0] = random.uniform(-5, 5)    # xs
        points[i][1] = random.uniform(-5, 5)    # ys
        points[i][2] = random.uniform(5, 100)     # zs

    # store the coordinates of the points visible on both images.
    img1xcoords = []
    img1ycoords = []
    img2xcoords = []
    img2ycoords = []

    counter = 0   # how many matched points are there? 

    for point in points:
        # is it seen by camera 1?
        seen1, x1, y1 = pointInCamera1Image(point, sensor1_width, sensor1_height, focal_length1, pixel_size1, camera1centre)

        # is it seen by camera 2? 
        seen2, x2, y2 = pointInCamera2Image(point, sensor2_width, sensor2_height, focal_length2, pixel_size2, camera2centre, tvec, R)  # TBD

        # need to be seen by both cameras to be a matched point...

        # print("Seen by camera 1? ", seen1)
        # print("Seen by camera 2? ", seen2 )

        # print(x1, y1)
        # print(x2, y2)

        if seen1 and seen2:  # seen by both cameras
            img1xcoords.append(x1)
            img1ycoords.append(y1)
            img2xcoords.append(x2)
            img2ycoords.append(y2)
            counter += 1

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

    fig = plt.figure()
    ax1 = plt.subplot(121)
    plt.plot(img1xcoords, img1ycoords, "bx")
    # plot the sensor on
    plt.plot([0,w1], [0, 0], "r", alpha=0.3)
    plt.plot([w1,w1], [0, h1], "r", alpha=0.3)
    plt.plot([w1,0], [h1, h1], "r", alpha=0.3)
    plt.plot([0, 0], [h1, 0], "r", alpha=0.3)

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

    plt.title("Camera 2 Image")
    plt.xlabel("x-direction (pixels)")

    plt.suptitle("Points as imaged by two cameras")
    plt.show()

main()
    
    









