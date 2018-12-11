#!/usr/bin/env python
# coding: utf-8

# In[1]:


# necessary modules
import numpy as np      # for maths
import cv2 as cv        # for camera tools
import glob             
import os
import sys, time


# In[2]:


# termination criteria
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

cols, rows = 7, 10

shape = (cols, rows)


# In[3]:


# prepare object points
# prepare object points
objp = np.zeros((cols*rows,3), np.float32) # zero array for 8 x 11 circle board
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)  # format shape of array

# arrays to store object points and image points
# objpoints1 = []
# imgpoints1 = []
# objpoints2 = []
# imgpoints2 = []


# In[4]:


# read the image file(s)
images = glob.glob("*.jpg")

folder = "found_patterns"

# if not os.path.exists(folder):
#     os.mkdir(folder)
    
# path = "{}/{}".format(os.getcwd(), folder)

# In[ ]:


counter, success = 1, 0 
size = (cols,rows)   # (cols, rows)
startTime = time.time()

path = '{}/{}'.format(os.getcwd(),folder)

      
def findCorners(fname):

    objpoints1 = []
    imgpoints1 = []
    # full size image for best accuracy
    img = cv.imread(fname)
    reimg = img   # in case need to resize

    # scale factor
    factor = img.shape[1]/reimg.shape[1]

    # convert to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find checkerboard corners
    ret, centres = cv.findChessboardCorners(gray, size, 
                                        flags=cv.CALIB_CB_ADAPTIVE_THRESH
                                            + cv.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True: 
        objpoints1.append(objp)

        centres2 = centres
        centres2 = cv.cornerSubPix(gray, centres, (11,11), (-1,-1), 
                                    criteria)

        imgpoints1.append(centres2)

        # draw and display the patterns
        drawimg = cv.drawChessboardCorners(reimg, size, centres2/factor, ret)

        # cv.imshow("img", drawimg)        
        # cv.imwrite(os.path.join(path , '{}.png'.format(success)), drawimg)

        # cv.waitKey(200)

    else:
        sys.stdout.write("Pattern not found...")
            
    cv.destroyAllWindows()

    coords = np.zeros((len(imgpoints1[0]),3))   # set up homogeneous coordinates 
    coords[:,-1] = 1

    for i in range(len(imgpoints1[0])):
        coords[i][:2] = imgpoints1[0][i]


    return coords

imgpoints1 = findCorners(images[0])
imgpoints2 = findCorners(images[1])


np.savez("imgpoints.npz", imgpoints1=imgpoints1, imgpoints2=imgpoints2)





        
