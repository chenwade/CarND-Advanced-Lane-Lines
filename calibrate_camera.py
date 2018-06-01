import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os

def calibrate_camera():

    if os.path.isfile("camera_cal/camera_coeff.p"):
        return

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # The images may have different detected checker board dimensions!
    # Currently, possible dimension combinations are: (9,6), (8,6), (9,5), (9,4) and (7,6)
    objp1 = np.zeros((6 * 9, 3), np.float32)  # 3d points in real world space
    objp1[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)   # 2d points in image plane.
    objp2 = np.zeros((6 * 8, 3), np.float32)
    objp2[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    objp3 = np.zeros((5 * 9, 3), np.float32)
    objp3[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
    objp4 = np.zeros((4 * 9, 3), np.float32)
    objp4[:, :2] = np.mgrid[0:9, 0:4].T.reshape(-1, 2)
    objp5 = np.zeros((6 * 7, 3), np.float32)
    objp5[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objp6 = np.zeros((6 * 5, 3), np.float32)
    objp6[:, :2] = np.mgrid[0:5, 0:6].T.reshape(-1, 2)

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_height = img.shape[0]
        image_width = img.shape[1]

        # Find the chessboard corners using possible combinations of dimensions.
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        objp = objp1
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
            objp = objp2
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (9, 5), None)
            objp = objp3
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (9, 4), None)
            objp = objp4
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            objp = objp5
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, (5, 6), None)
            objp = objp6
        # print("corners: ", corners.shape, "\n", corners)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners

            cv2.drawChessboardCorners(img, (corners.shape[1],corners.shape[0]), corners, ret)
#            write_name = 'corners_found'+str(idx)+'.jpg'
#            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (image_height, image_width), None, None)

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




