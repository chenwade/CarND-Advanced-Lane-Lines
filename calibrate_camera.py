import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os

def calibrate_camera():

    if os.path.isfile("camera_cal/camera_coeff.p"):
        return

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0), the three dimensional points
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners

            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            write_name = 'corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    # Do camera calibration given object points and image points
    image_size = (1280, 720)  #the size of the calibrate image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["objpoints"] = objpoints
    dist_pickle["imgpoints"] = imgpoints
    pickle.dump(dist_pickle, open("camera_cal/camera_coeff.p", "wb"))



if __name__ == "__main__":

    calibrate_camera()
    #undistorted the image
    dist_pickle = pickle.load(open("camera_cal/camera_coeff.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    img = cv2.imread('camera_cal/calibration11.jpg')
    undistored_img = cv2.undistort(img, mtx, dist, None, mtx)
    #show the original image and undistorted image
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(undistored_img)
    ax2.set_title('Undistorted Image', fontsize=10)
    plt.show()




