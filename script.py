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
import re

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

IMAGE_SHAPE = (1280, 1024)


@dataclass
class Checkboard:
    X_CONST: int = 6  # number of corners in X direction
    Y_CONST: int = 8  # number of corners in Y direction


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


SQUARE_SIZE = [28.37, 28.37, 1]



def get_image_number(image_name):
    try:
        return re.search(".*_(\d{1,}).png", image_name).group(1)
    except:
        raise (Exception(f"Could not get number from {image_name}"))


def cleanup():
    final_set = set()

    for index, camera_path in enumerate(camera_paths.values()):
        current_set = set([get_image_number(image_path.name) for image_path in (camera_path / 'corners').glob("*png")])
        if index == 0:
            final_set = current_set
        else:
            final_set = final_set.intersection(current_set)

    for camera_path in camera_paths.values():
        for photo in camera_path.glob("*png"):
            if get_image_number(photo.name) not in final_set:
                os.remove(photo)


def construct_objp(checkboard: Checkboard, square_size=SQUARE_SIZE):
    objp = np.zeros((checkboard.X_CONST * checkboard.Y_CONST, 3), np.float32)
    objp[:, :2] = np.mgrid[:checkboard.X_CONST, :checkboard.Y_CONST].T.reshape(-1, 2)
    return (objp * SQUARE_SIZE).astype(np.float32)


def calibrate_camera(camera_paths, objp):
    for camera_name, camera_path in camera_paths.items():
        print(camera_name)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = [path for path in camera_path.glob("*.png")]
        index = -1
        for fname in tqdm.tqdm(images):
            index += 1
            try:
                img = imread(fname.__str__())
                gray_img = cvtColor(img, cv.COLOR_BGR2GRAY)
            except:
                print("Skipping due to read error")
                os.remove(fname)
                continue
            ret, corners = cv.findChessboardCorners(gray_img, (checkboard_params.X_CONST, checkboard_params.Y_CONST),
                                                    None)
            if ret:
                if corners[0, 0, 1] < corners[-1, 0, 1]:
                    # we only want files in one direction
                    print("Incorrect orientation, skipping")
                    os.remove(fname)
                    continue
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (checkboard_params.X_CONST, checkboard_params.Y_CONST), corners2, ret)
                plt.imshow(img)
                destination = camera_path / 'corners' / os.path.basename(fname)
                print(f"Saving to {destination}")
                plt.savefig(destination)
                plt.close()
            else:
                print("Chessboard not found, skipping")
                os.remove(fname)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)
        calibration_results = CalibrationResults(ret, mtx, dist, rvecs, tvecs)

        camera_pikle_path = camera_path / 'calibration_results' / camera_name

        with open(camera_pikle_path, 'wb') as file:
            print(f"Saving camera to {camera_pikle_path}")
            pickle.dump(calibration_results, file)

        imgpoints_path = camera_path / 'calibration_results' / 'img_points'
        with open(imgpoints_path, 'wb') as file:
            print(f"Saving imgpoints to {imgpoints_path}")
            pickle.dump(imgpoints, file)

        objpoints_path = camera_path / 'calibration_results' / 'obj_points'
        with open(objpoints_path, 'wb') as file:
            print(f"Saving objpoints to {objpoints_path}")
            pickle.dump(objpoints, file)


def remove_distortion(calibration_results, image):
    h, w = image.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(calibration_results.mtx, calibration_results.dist, (w, h), 1,
                                                     (w, h))
    dst = cv.undistort(image, calibration_results.mtx, calibration_results.dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]  # image with removed distortions
    return dst, newcameramtx


def undistort_images(camera_paths):
    for camera_name, camera_path in camera_paths.items():
        with open(camera_path / 'calibration_results' / camera_name, 'rb') as file:
            calibration_results = pickle.load(file)

        images = (camera_path).glob("*.png")
        for fname in tqdm.tqdm(images):
            img = imread(fname.__str__())
            dst, newcameramtx = remove_distortion(calibration_results, img)
            plt.imshow(dst)
            destination = camera_path / 'undistorted' / os.path.basename(fname)
            plt.savefig(destination)
            plt.close()
        newcameramtx_path = camera_path / 'calibration_results' / f"{camera_name}_undistorted"
        with open(newcameramtx_path, 'wb') as file:
            print(f'Saving new camera matrix to {newcameramtx_path}')
            pickle.dump(newcameramtx, file)

def stereo_calibrate(dims, objpoints, imgpoints_l, imgpoints_r, M1, d1, M2, d2):
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    flags |= cv.CALIB_USE_INTRINSIC_GUESS
    flags |= cv.CALIB_FIX_FOCAL_LENGTH
    flags |= cv.CALIB_ZERO_TANGENT_DIST

    stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
    ret, M1, d1, M2, d2, R, T, E, F = cv.stereoCalibrate( objpoints, imgpoints_l, imgpoints_r, M1, d1, M2, d2, dims, criteria=stereocalib_criteria, flags=flags)

    camera_model = {
        "M1": M1.tolist(),
        "d1": d1.tolist(),
        "M2": M2.tolist(),
        "d2": d2.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist()
    }

    cv.destroyAllWindows()
    return camera_model


def rectify():
    from gems.io import Pickle

    print("Loading")

    left_camera = Pickle.load(camera_paths['left'] / 'calibration_results/left')
    right_camera = Pickle.load(camera_paths['right'] / 'calibration_results/right')

    checkboard_params = Checkboard(6,8)
    objp = construct_objp(checkboard_params)

    left_path_imgpoints = calibration_path / 'left/calibration_results/img_points'
    right_path_imgpoints = calibration_path / 'right/calibration_results/img_points'

    left_imgpoints = Pickle.load(left_path_imgpoints)
    right_imgpoints = Pickle.load(right_path_imgpoints)

    object_points = [objp for i in range(len(left_imgpoints))]

    print("Stereocalibrating and Stereorecrifying")


    camera_model = stereo_calibrate(IMAGE_SHAPE, object_points, left_imgpoints, right_imgpoints, left_camera.mtx, left_camera.dist, right_camera.mtx, right_camera.dist)
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(np.array(camera_model['M1']), np.array(camera_model['d1']), np.array(camera_model['M2']), np.array(camera_model['d2']), IMAGE_SHAPE, np.array(camera_model['R']), np.array(camera_model['T']))


    print("Remaping")

    map1x, map1y = cv.initUndistortRectifyMap(
        cameraMatrix=np.array(camera_model['M1']),
        distCoeffs=np.array(camera_model['d1']),
        R=R1,
        newCameraMatrix=P1,
        size=(1280, 1024),
        m1type=cv.CV_32FC1)

    map2x, map2y = cv.initUndistortRectifyMap(
        cameraMatrix=np.array(camera_model['M2']),
        distCoeffs=np.array(camera_model['d2']),
        R=R2,
        newCameraMatrix=P2,
        size=(1280, 1024),
        m1type=cv.CV_32FC1)

    print("rectifing")

    for filename in (camera_paths['left']).glob("*.png"):
        number = get_image_number(filename.name)
        left_img = cv.imread(filename.__str__())
        right_img = cv.imread((camera_paths['right'] / f'right_{number}.png').__str__())

        imgl_rect = cv.remap(left_img, map1x, map1y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
        imgr_rect = cv.remap(right_img, map2x, map2y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

        fig, ax = plt.subplots(nrows=1, ncols=2)

        plt.subplot(1, 2, 1)
        plt.imshow(imgl_rect)

        plt.subplot(1, 2, 2)
        plt.imshow(imgr_rect)

        outpath = f'./outputs/rectified/{number}.png'
        print(f"Saving rectified image to {outpath}")
        plt.savefig(outpath)
        plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default='advanced-computer-vision/data/calibration')
    parser.add_argument('-c', '--calibrate', action='store_true')
    parser.add_argument('-u', '--undistort', action='store_true')
    parser.add_argument('-cl', '--clean', action='store_true')
    parser.add_argument('-r', '--rectify', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')



    args = parser.parse_args()

    calibration_path = Path(args.input_path)

    camera_paths = {
        'left': calibration_path / 'left',
        'right': calibration_path / 'right',
    }


    if args.all:
        checkboard_params = Checkboard(6, 8)
        objp = construct_objp(checkboard_params)
        calibrate_camera(camera_paths, objp)
        cleanup()
        calibrate_camera(camera_paths, objp)
        undistort_images(camera_paths)
        rectify()

    if args.clean:
        cleanup()

    if args.calibrate:
        checkboard_params = Checkboard(6, 8)
        objp = construct_objp(checkboard_params)
        calibrate_camera(camera_paths, objp)

    if args.undistort:
        undistort_images(camera_paths)

    if args.rectify:
        rectify()
