import cv2  as cv
import numpy as np
import os
import glob
import shutil
import pyfiglet


print(pyfiglet.figlet_format("Welcome  to Camera Calibration"))
print('\n Press q to end the camera grab \n \n ')
cam = cv.VideoCapture('/dev/video0')
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboardSize = (9,7)
frameSize = (640,480)
i = 0
if not os.path.isfile('Images'):
    os.mkdir('Images')
    os.chdir('Images')
else:
    print('Images folder already exist \n')
    os.chdir('Images')


while True:
    ret, frame = cam.read()
    if ret:
        cv.imshow('picam',frame)
        cv.moveWindow('picam',0,0)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9,7), None)
        if ret == True:
            i+=1
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            draw_chess_pattern = frame.copy()
            cv.drawChessboardCorners(draw_chess_pattern, (9,7), corners2, ret)
            cv.imshow('checkerboard pattern', draw_chess_pattern)
            cv.imwrite('frame'+str(i)+'.jpg',frame)



        if cv.waitKey(1)==ord('q'):
            break
    
    else:
        print(pyfiglet.figlet_format("Problem Fetching Camera"))
        exit()

cam.release()
cv.destroyAllWindows()


os.chdir('..')
print('Current working directory: ', os.listdir())
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Images/*.jpg')
print(len(images))
i = 0
for image in images:
    i += 1
    img = cv.imread(image)
    print("Images Processed: ",(round(i/len(images))*100, 2))
    os.system('clear')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('calibrating image', gray)
    cv.waitKey(50)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(50)



cv.destroyAllWindows()


print(pyfiglet.figlet_format("Calibration In Progress"))
print('\U0001F62A')
############## CALIBRATION #######################################################
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
np.savez("parameters.npz",mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
np.savez("points.npz",objpoints=objpoints,imgpoints=imgpoints)
############## UNDISTORTION #####################################################

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "\n \n Total error: {}".format(mean_error/len(objpoints)) )

shutil.rmtree('Images')
print('\n \n Calibration values \n ',mtx,'\n', ret, '\n', dist,'\n')
print('Data folder deleted \n')
print('\n \n All calibration parameters saved  \U0001f600 \n \n')

