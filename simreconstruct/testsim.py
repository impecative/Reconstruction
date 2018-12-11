from __future__ import division, print_function # Python 2 compatibility
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random

# determine the equation of a line from point P=(x1, y1, z1) to point Q=(x2, y2, z2).
# direction vector d = PQ = (x2-x1, y2-y1, z2-z1) = (l, m, n)
# then any point on the line must satisfy: 
# (x-x1)/l = (y-y1)/m = (z-z1)/n
# to evaluate x,y coordinates on image, we need to know where the image plane sits in space (at z=f (focal length) from camera centre)

def pointInFrame(pointcoordinate, cameracentre, sensor_width, sensor_height, focal_length):
    '''Use np.array() for coordinates...'''
    w, h = sensor_width, sensor_height
    f = focal_length

    # extract the coordinates
    x1, y1, z = pointcoordinate         # in units of metres
    x2, y2, z2 = cameracentre           # in units of metres

    # first check whether the point is in front of the camera image plane
    if not z >= z2 + f:
        return False


    lTop    = np.array([x2-w/2, y2+h/2, z2+f])    # left top corner of sensor
    lBottom = np.array([x2-w/2, y2-h/2, z2+f])    # left bottom corner of sensor
    rTop    = np.array([x2+w/2, y2+h/2, z2+f])    # right top corner of sensor
    rBottom = np.array([x2+w/2, y2-h/2, z2+f])    # right bottom corner of sensor

    # make a list of direction vectors
    dvectors = []

    # direction vectors to sensor corners
    dlTop    = lTop - cameracentre
    dlBottom = lBottom - cameracentre
    drTop    = rTop - cameracentre
    drBottom = rBottom - cameracentre

    # for top right:
    l, m, n = drTop 
    assert (l != 0) and (m != 0) and (n != 0), "direction vector drTop has zeros in it..."
    ymax = m/n * (z-(z2+f)) + rTop[1] 
    xmax = l/n * (z-(z2+f)) + rTop[0] 

    # bottom left
    l, m, n = dlBottom
    assert (l != 0) and (m != 0) and (n != 0), "direction vector dlBottom has zeros in it..."
    ymin = m/n * (z-(z2+f)) + lBottom[1]
    xmin = l/n * (z-(z2+f)) + lBottom[0] 

    if (x1 <= xmax and x1 >= xmin) and (y1 <= ymax and y1 >= ymin):
        return True
    else:
        return False


def getImageCoordinates(pointcoordinate, cameracentre, focal_length, pixel_size):
    # ASSUME pixels are square! 
    # This will need to all be in the same units of pixels, so pixel dimensions are also needed...
    x1, y1, z1 = pointcoordinate    # in metres
    x2, y2, z2 = cameracentre       # in metres.

    #  convert to pixel units
    x1, y1, z1 = x1/pixel_size, y1/pixel_size, z1/pixel_size
    x2, y2, z2 = x2/pixel_size, y2/pixel_size, z2/pixel_size

    # need focal length in pixels.
    f = focal_length / pixel_size

    # direction vector:
    d = (x2-x1, y2-y1, z2-z1)    # d = (l, m, n)
    l, m, n = d
    # print(l, m, n)

    if (l != 0) and (m != 0) and (n != 0):
        y = m/n * (focal_length+z2-z1) + y1
        x = l/n * (focal_length+z2-z1) + x1

        # in pixel coordinates: 
        x, y = x/pixel_size, y/pixel_size
        return x,y

    elif l == 0:
        x = x1
        if m == 0:
            y = y1
            x, y = x/pixel_size, y/pixel_size
            return x, y  # image coordinates in pixels.   # DOES THIS FOLLOW IMAGE COORDINATE CONVENTIONS? NO...

def RotationMatrix(yaw, pitch, roll):
    R_x = np.array([[1,0,0], [0, np.cos(np.deg2rad(roll)), -np.sin(np.deg2rad(roll))], [0, np.sin(np.deg2rad(roll)), np.cos(np.deg2rad(roll))]])
    R_y = np.array([[np.cos(np.deg2rad(pitch)), 0, np.sin(np.deg2rad(pitch))],[0,1,0], [-np.sin(np.deg2rad(pitch)), 0, np.cos(np.deg2rad(pitch))]])
    R_z = np.array([[np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw)), 0], [np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw)),0], [0,0,1]])

    R = np.linalg.multi_dot([R_z, R_y, R_x])

    return R

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

f = 50e-1 # (50mm focal length)
w,h = 23.5e-3, 15.6e-3 # sensor width and height
cameracentre1 = np.array([0,0,0])
cameracentre2 = np.array([0, 0, 50])
point = np.array([0, 0, 25])

# four corners of rectangle (sensor) for cam 1
# take the normal to be the z axis, i.e z is constant f for all corners.
BL = np.array([-w/2, -h/2, f]) - cameracentre1      # bottom left point
TL = np.array([-w/2, h/2, f]) - cameracentre1       # top left point
TR = np.array([w/2, h/2, f]) - cameracentre1        # top right point
BR = np.array([w/2, -h/2, f]) - cameracentre1       # bottom right point


# rotate the points
R = RotationMatrix(yaw=0, pitch=180, roll=0)   # all in degrees
BL2 = R @ BL
TL2 = R @ TL
TR2 = R @ TR
BR2 = R @ BR

# translate the points
tvec = cameracentre2 - cameracentre1
BL2 = BL2 + tvec
TL2 = TL2 + tvec
TR2 = TR2 + tvec
BR2 = BR2 + tvec

distanceTop = TL2 - BL2
distanceTop  = np.linalg.norm(distanceTop)

assert np.allclose(h, distanceTop), "Sensor size of second camera has gone wrong!!!"

# define the plane of the second sensor from any of the three points... (they're all coplanar, so it doesn't matter which ones...)
A, B, C = BL2, TL2, TR2 
normal = np.cross((B-A), (C-B))
# scalar component:
d = np.dot(normal, A)

a,b,c = normal

# check that the final point lies on the plane...
val = normal[0]*BR2[0] + normal[1]*BR2[1] + normal[2]*BR2[2]

print("The equation of the plane is {}x + {}y + {}z = {}".format(a,b,c,d))

assert np.allclose(d,val), "The final point doesn't lie in the plane of the sensor?? Check maths..."

# where does the line linking the point and the 3D point intersect the plane?
# Let A = 3D point, B = camera 2 centre
A, B = point, cameracentre2
AB = cameracentre2 - point
t = (d - a*A[0] - b*A[1] - c*A[2])/(a*AB[0] + b*AB[1] + c*AB[2])
intersection = A + (B-A)*t

# translate the intersection back to the original camera... this approach is only valid if the cameras are identical...
newintersection = intersection - tvec
newintersection = np.linalg.inv(R) @ newintersection

# can this point be seen? 
minx, miny, _ = BL
maxx, maxy, _ = TR

x,y,_ = newintersection

pointcanbeseen = False

if (x >= minx and x <= maxx) and (y >= miny and y <= maxy):
    pointcanbeseen = True
else:
    pass

print("The point can be seen: ", pointcanbeseen)







ax.scatter(intersection[0], intersection[1], intersection[2], c="m")
ax.scatter(newintersection[0], newintersection[1], newintersection[2], c="y")
ax.scatter(point[0], point[1], point[2], c="k")
# ax.plot([point[0], cameracentre2[0]], [point[1], cameracentre2[1]], [point[2], cameracentre2[2]], c="g")









ax.scatter(cameracentre1[0], cameracentre1[1], cameracentre1[2], c="r", label="Camera 1") # draw camera 1 
ax.scatter(cameracentre2[0], cameracentre2[1], cameracentre2[2], c="g", label="Camera 2") # draw camera 2

ax.plot([BL[0], TL[0]], [BL[1], TL[1]], [BL[2], TL[2]], c="b")
ax.plot([TL[0], TR[0]], [TL[1], TR[1]], [TL[2], TR[2]], c="b")
ax.plot([TR[0], BR[0]], [TR[1], BR[1]], [TR[2], BR[2]], c="b")
ax.plot([BR[0], BL[0]], [BR[1], BL[1]], [BR[2], BL[2]], c="b")

ax.plot([BL2[0], TL2[0]], [BL2[1], TL2[1]], [BL2[2], TL2[2]], c="b")
ax.plot([TL2[0], TR2[0]], [TL2[1], TR2[1]], [TL2[2], TR2[2]], c="b")
ax.plot([TR2[0], BR2[0]], [TR2[1], BR2[1]], [TR2[2], BR2[2]], c="b")
ax.plot([BR2[0], BL2[0]], [BR2[1], BL2[1]], [BR2[2], BL2[2]], c="b")



ax.set_xlabel("x ")
ax.set_ylabel("y ")
ax.set_zlabel("z ")
ax.legend()
plt.show()







# test. 
# If 3D point is at position (5, 10, 15) and camera is at origin
# For a focal length of 5 pixels, what is the coordinates on the image projection? 


"""points3D = np.zeros((5,3))

for i in range(points3D.shape[0]):
    # points3D[i][3] = 1   # for homogeneous coordinates
    points3D[i][0] = random.uniform(-19,40)   # x
    points3D[i][1] = random.uniform(-5,5)    # y
    points3D[i][2] = random.uniform(-20, 50)  # z

sensor_width, sensor_height = 23.5e-3, 15.6e-3 # metres
f = 50e-3 # focal length in m
cameracentre = np.array([0,0,0])
pixel_size = 3.9e-6
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs, ys, zs = points3D[:,0], points3D[:,1], points3D[:,2]
seecounter = 0
cantseecounter = 0
imgcoords = []

for x,y,z in zip(xs, ys, zs):
    canSee = pointInFrame(np.array([x,y,z]), cameracentre, sensor_width, sensor_height, focal_length=f)
    if canSee:
        imgx, imgy = getImageCoordinates(np.array([x,y,z]), cameracentre, f, pixel_size)
        imgcoords.append((imgx,imgy))
        ax.plot([x,cameracentre[0]], [y,cameracentre[1]], [z,cameracentre[2]], c="g")
        ax.scatter(x, y, z, c="b")
        seecounter += 1
    else:
        ax.plot([x,cameracentre[0]], [y,cameracentre[1]], [z,cameracentre[2]], c="r")
        ax.scatter(x,y,z, c="m")
        cantseecounter += 1
        pass



print("Can see {} points".format(seecounter))
print("Can't see {} points".format(cantseecounter))


ax.scatter(cameracentre[0], cameracentre[1], cameracentre[2], c="r", label="Centre of first camera")




# sensor width, height = (23.5 x 15.6) e-3 m 
# pixel size = 3.9 e-6 m


ax.plot([cameracentre[0]-sensor_width/2,cameracentre[0]-sensor_width/2], [cameracentre[1]-sensor_height/2, cameracentre[1]+sensor_height/2], [cameracentre[2]+f, cameracentre[2]+f],  c="m")
ax.plot([cameracentre[0]-sensor_width/2,cameracentre[0]+sensor_width/2], [cameracentre[1]+sensor_height/2, cameracentre[1]+sensor_height/2], [cameracentre[2]+f,cameracentre[2]+f],  c="m")
ax.plot([cameracentre[0]+sensor_width/2,cameracentre[0]+sensor_width/2], [cameracentre[1]+sensor_height/2, cameracentre[1]-sensor_height/2], [cameracentre[2]+f,cameracentre[2]+f],  c="m")
ax.plot([cameracentre[0]+sensor_width/2,cameracentre[0]-sensor_width/2], [cameracentre[1]-sensor_height/2, cameracentre[1]-sensor_height/2], [cameracentre[2]+f,cameracentre[2]+f], c= "m")


ax.set_xlabel("x ")
ax.set_ylabel("y ")
ax.set_zlabel("z ")

# plt.show()


# print(imgcoords)

# 2D figure of image
plt.figure()

for i in range(len(imgcoords)):
    x, y = imgcoords[i][:]
    # print(x,y)
    plt.plot(x,y, "bx")

# draw the sensor on (in pixels)
w, h =sensor_width/pixel_size, sensor_height/pixel_size  # in pixels

plt.plot([w/2, w/2], [h/2, -h/2], "r")
plt.plot([w/2, -w/2], [-h/2, -h/2], "r")
plt.plot([-w/2, -w/2], [-h/2, h/2], "r")
plt.plot([-w/2, w/2], [h/2, h/2], "r")

plt.xlim(xmin=-w/2-w/8, xmax=w/2 + w/8)
plt.ylim(ymin=-h/2-h/8, ymax=h/2+h/8)


plt.show()
# plt.draw()
# plt.pause(1) # <-------
# input("<Hit Enter To Close>")
# plt.close(fig)"""









