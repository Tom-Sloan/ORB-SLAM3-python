import time
import orbslam3
import argparse
from glob import glob
import os 
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_file", required=True)
parser.add_argument("--settings_file", required=True)
parser.add_argument("--path_to_images", required=True)
args = parser.parse_args()

# Get all PNG files in the rgb/ subdirectory of the dataset path
# - glob() finds all files matching the pattern 'rgb/*.png'
# - os.path.join() combines dataset_path with 'rgb/*.png' using proper path separator
# - sorted() ensures files are in alphabetical/numerical order
# - Result is a list of full paths to all RGB image files
img_files = sorted(glob(os.path.join(args.path_to_images, 'data/*.png')))
slam = orbslam3.system(args.vocab_file, args.settings_file, orbslam3.Sensor.MONOCULAR)
slam.set_use_viewer(False)
slam.initialize()

start_time = time.time()
frame_count = 0

for img in img_files:
    timestamp = img.split('/')[-1][:-4]
    img = cv2.imread(img, -1)
    pose = slam.process_image_mono(img, float(timestamp))
    trajectory = slam.get_trajectory()
    frame_count += 1

elapsed_time = time.time() - start_time
fps = frame_count / elapsed_time
print(f"\nProcessed {frame_count} frames in {elapsed_time:.2f} seconds")
print(f"Average FPS: {fps:.2f}")