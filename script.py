from xmlrpc.server import MultiPathXMLRPCServer
import tqdm
import pickle
from cv2 import cornerSubPix, imread, cvtColor, imshow, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS
import cv2 as cv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
import argparse

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

@dataclass
class Checkboard:

    X_CONST: int = 7 # number of corners in X direction
    Y_CONST: int = 10 # number of corners in Y direction


@dataclass
class CalibrationResults:

    ret = None
    mtx = None
    dist = None
    rvecs = None
    tvecs = None

    def __init__(self, ret, mtx, dist, rvecs, tvecs):
        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs


CONST_SIZE = 5

calibration_path = Path('advanced-computer-vision/data/calibration')

camera_paths = {
    'cam1' : calibration_path / 'cam1',
    'cam2' : calibration_path / 'cam2',
    'cam3' : calibration_path / 'cam3',
    'cam4' : calibration_path / 'cam4',
}

def construct_objp(checkboard: Checkboard, square_size = CONST_SIZE):
    objp = np.zeros((checkboard.X_CONST * checkboard.Y_CONST,3), np.float32)
    objp[:, :2] = np.mgrid[:checkboard.X_CONST, :checkboard.Y_CONST].T.reshape(-1,2)
    return objp * CONST_SIZE

def calibrate_camera(camera_paths, objp):

    for camera_name, camera_path in camera_paths.items():

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = camera_path.glob("*.png")
        for fname in tqdm.tqdm(images):
            img = imread(fname.__str__())
            gray_img = cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray_img, (checkboard_params.X_CONST,checkboard_params.Y_CONST), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray_img,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (checkboard_params.X_CONST,checkboard_params.Y_CONST), corners2, ret)
                plt.imshow(img)
                destination = camera_path / 'corners' / os.path.basename(fname)
                plt.savefig(destination)
                plt.close()
        

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,imgpoints,gray_img.shape[::-1],None,None)
        calibration_results = CalibrationResults(ret, mtx, dist, rvecs, tvecs)

        with open(camera_path / 'calibration_results' / camera_name, 'wb') as file:
            pickle.dump(calibration_results, file)

def undistort_images(camera_paths):

    for camera_name, camera_path in camera_paths.items():
        with open(camera_path / 'calibration_results' / camera_name, 'rb') as file:
            calibration_results = pickle.load(file)

        images = (camera_path).glob("*.png")
        for fname in tqdm.tqdm(images):
            img = imread(fname.__str__())
            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(calibration_results.mtx, calibration_results.dist, (w,h), 1, (w,h))
            dst = cv.undistort(img, calibration_results.mtx, calibration_results.dist, None, newcameramtx)
            x, y, w, h = roi
            print(roi)
            dst = dst[y:y+h, x:x+w] # image with removed distortions
            plt.imshow(dst)
            destination = camera_path / 'undistorted' / os.path.basename(fname)
            plt.savefig(destination)
            plt.close()        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calibrate', action='store_true')
    parser.add_argument('-u', '--undistort', action='store_true')

    args = parser.parse_args()

    if args.calibrate:
        checkboard_params = Checkboard(7,10)
        objp = construct_objp(checkboard_params)
        calibrate_camera(camera_paths, objp)

    if args.undistort:
        undistort_images(camera_paths)