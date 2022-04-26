import argparse
import os
import shutil

from pathlib import Path
import pickle
import numpy as np

from script import CalibrationResults # must be important to unpickle files
from prepare_structure import Branch

def calculate_distance_beteen_cameras(tvecs):
    first = (np.array(tvecs[0]).squeeze())
    second = (np.array(tvecs[1]).squeeze())
    dist = np.sqrt(np.sum((first - second)**2)) / len(first)
    return dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default='advanced-computer-vision/data/calibration')
    parser.add_argument('-o', '--output_path', default='./output')
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    calibration_path = Path(args.input_path)
    output_path = Path(args.output_path)
    clean = args.clean
    
    if clean:
        shutil.rmtree(output_path)       
    os.makedirs(output_path, exist_ok=True)
    
    left_branch = Branch(calibration_path, 'left')
    right_branch = Branch(calibration_path, 'right')

    branches = [left_branch, right_branch]

    tvecs = []

    for branch in branches:
        branch_path = branch.folder_path / branch.name /'calibration_results'
        _output_path = str(output_path / branch.name)
        
        # calibration results
        path_calibration_results = branch_path / branch.name
        with open(path_calibration_results, 'rb') as file:
            camera = pickle.load(file)
            tvecs.append(camera.tvecs)
            with open(f'{_output_path}_camera.txt','w') as file_out:
                file_out.write(str(camera.__dict__))

        # object points
        path_objpoints =  branch_path / "obj_points"
        with open(path_objpoints, 'rb') as file:
            obj_points = pickle.load(file)
            with open(f'{_output_path}_objpoints.txt', 'w') as file_out:
                file_out.write(str({'points':obj_points}))
                
        # image points   
        path_imgpoints  = branch_path / "img_points" 
        with open(path_imgpoints, 'rb') as file:
            imgpoints = pickle.load(file)
            with open(f'{_output_path}_imgpoints.txt', 'w') as file_out:
                file_out.write(str({'points':imgpoints}))

        # undistorted camera matrix
        path_undistorted_camera_matrix  = branch_path / f'{branch.name}_undistorted'
        with open(path_undistorted_camera_matrix, 'rb') as file:
            camera = pickle.load(file)
            with open(f'{_output_path}_undistorted_camera.txt','w') as file_out:
                file_out.write(str(camera))
        
        with open(f'{_output_path}_files.txt', 'w') as file:
            file.write('\n'.join([photo.name for photo in path_calibration_results.glob('*.png')]))

    dist = calculate_distance_beteen_cameras(tvecs)
    with open(f'{output_path}/dist_in_mm.txt', 'w') as file:
        file.write(str(dist))
        


        