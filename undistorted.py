import pickle
import cv2
import numpy as np

import glob
# Read in the saved objpoints and imgpoints
#file_list = 
objpoints = []
imgpoints = []

objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

file_list = glob.glob('./calibration*.jpg')

for idx, fname in enumerate(file_list):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,(9,6),None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        cv2.drawChessboardCorners(img, (9,6),corners,ret)
        put_name = 'corners_labeled'+str(idx)+'.jpg'
        cv2.imwrite(put_name,img)


    # Use one image to calibrate the camera 

img = cv2.imread('./calibration3.jpg')
img_size = (img.shape[1],img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, \
    imgpoints, img.shape[0:2],None,None)

#pickle the results

dic_pickle = {}
dic_pickle['mtx'] = mtx
dic_pickle['dist'] = dist
pickle.dump(dic_pickle,open('./calibration.p','wb'))

