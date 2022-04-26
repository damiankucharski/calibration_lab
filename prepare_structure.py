from pathlib import Path
import argparse
import os
import shutil
from dataclasses import dataclass


def make_folders(output_path, clear = False):
    paths = [output_path / 'left', output_path / 'right']

    for path in paths:
        if clear:
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        os.makedirs(path / 'calibration_results')
        os.makedirs(path/ 'corners')
        os.makedirs(path/ 'undistorted')
    

def copy_images(branches, dest):
    
    for branch in branches:
        for file in branch.files:
            shutil.copy(file, dest / branch.name)


@dataclass
class Branch:
    folder_path: Path
    name: str
    files: list = None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default = 'advanced-computer-vision/data/calibration')
    parser.add_argument('-o', "--output_path", default = 'advanced-computer-vision/data/divided_images')
    parser.add_argument('-c', '--clear', action='store_true')

    args = parser.parse_args()

    calibration_path = Path(args.input_path)
    output_path = Path(args.output_path)
    clear = args.clear

    left = calibration_path.glob("left_*.png")
    right = calibration_path.glob("right_*.png")

    left_branch = Branch(calibration_path, 'left', left)
    right_branch = Branch(calibration_path, 'right', right)



    make_folders(output_path,clear)
    copy_images([left_branch, right_branch], output_path)

    

    