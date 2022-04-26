from pathlib import Path
import pickle
import numpy as np

from script import CalibrationResults

calibration_path = Path('advanced-computer-vision/data/calibration')


left_path_camera = calibration_path / 'left/calibration_results/left'
left_path_imgpoints = calibration_path / 'left/calibration_results/img_points'
left_path_objpoints = calibration_path / 'left/calibration_results/obj_points'

right_path_camera = calibration_path / 'right/calibration_results/right'
right_path_imgpoints = calibration_path / 'right/calibration_results/img_points'
right_path_objpoints = calibration_path / 'right/calibration_results/obj_points'

with open(left_path_camera, 'rb') as file:
    left_camera = pickle.load(file)
    with open('./outputs/left_camera.txt','w') as file_out:
        file_out.write(str(left_camera.__dict__))

with open(right_path_camera, 'rb') as file:
    right_camera = pickle.load(file)
    with open('./outputs/right_camera.txt', 'w') as file_out:
        file_out.write(str(right_camera.__dict__))

with open(left_path_objpoints, 'rb') as file:
    left_obj_points = pickle.load(file)
    with open('./outputs/left_objpoints.txt', 'w') as file_out:
        file_out.write(str({'points':left_obj_points}))

with open(right_path_objpoints, 'rb') as file:
    right_obj_points = pickle.load(file)
    with open('./outputs/right_objpoints.txt', 'w') as file_out:
        file_out.write(str({'points':right_obj_points}))

with open(left_path_imgpoints, 'rb') as file:
    left_imgpoints = pickle.load(file)
    with open('./outputs/left_imgpoints.txt', 'w') as file_out:
        file_out.write(str({'points':left_imgpoints}))

with open(right_path_imgpoints, 'rb') as file:
    right_imgpoints = pickle.load(file)
    with open('./outputs/right_imgpoints.txt', 'w') as file_out:
        file_out.write(str({'points':right_imgpoints}))

with open('./outputs/left_files.txt', 'w') as file:
    path_left_files = calibration_path / 'left'
    file.write('\n'.join([photo.name for photo in path_left_files.glob('*.png')]))

with open('./outputs/right_files.txt', 'w') as file:
    path_left_files = calibration_path / 'right'
    file.write('\n'.join([photo.name for photo in path_left_files.glob('*.png')]))

first = (np.array(right_camera.tvecs).squeeze())
second = (np.array(left_camera.tvecs).squeeze())
dist = np.sqrt(np.sum((first - second)**2)) / len(first)

with open('./outputs/dist_in_mm.txt', 'w') as file:
    file.write(str(dist))


