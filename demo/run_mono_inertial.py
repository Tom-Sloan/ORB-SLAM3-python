import orbslam3
import argparse
from glob import glob
import os 
import cv2
import numpy as np
import csv
from collections import defaultdict
from orbslam3 import IMU

def load_images(image_path, timestamps_file):
    """Load images and their timestamps"""
    with open(timestamps_file, 'r') as f:
        timestamps = []
        image_files = []
        for line in f:
            if line[0] != '#':  # Skip comments
                line = line.strip()
                timestamp = float(line.split(',')[0]) / 1e9  # Convert ns to s
                image_name = line.split(',')[1]
                image_files.append(os.path.join(image_path, image_name))
                timestamps.append(timestamp)
    return image_files, timestamps

def load_imu(imu_path):
    """Load IMU measurements from CSV file"""
    timestamps = []
    accelerometer = []
    gyroscope = []
    
    with open(imu_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            # Convert from ns to s
            timestamp = float(row[0]) / 1e9
            # IMU data format: timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
            gyro = [float(row[1]), float(row[2]), float(row[3])]
            acc = [float(row[4]), float(row[5]), float(row[6])]
            
            timestamps.append(timestamp)
            gyroscope.append(gyro)
            accelerometer.append(acc)
            
    return timestamps, accelerometer, gyroscope

def get_imu_measurements_between(t0, t1, imu_timestamps, accelerometer, gyroscope):
    """Get all IMU measurements between two timestamps"""
    measurements = []
    for ts, acc_vals, gyro_vals in zip(imu_timestamps, accelerometer, gyroscope):
        if t0 < ts <= t1:
            point = IMU.Point(acc_vals[0], acc_vals[1], acc_vals[2],
                              gyro_vals[0], gyro_vals[1], gyro_vals[2],
                              ts)
            measurements.append(point)
    return measurements

def main():
    parser = argparse.ArgumentParser(description='Run ORB-SLAM3 Mono-Inertial')
    parser.add_argument('--vocab_file', required=True, help='Path to ORB vocabulary file')
    parser.add_argument('--settings_file', required=True, help='Path to settings file')
    parser.add_argument('--sequence_path', required=True, help='Path to EuRoC sequence folder')
    args = parser.parse_args()

    # Setup paths
    image_path = os.path.join(args.sequence_path, 'mav0/cam0/data')
    image_timestamps_file = os.path.join(args.sequence_path, 'mav0/cam0/data.csv')
    imu_file = os.path.join(args.sequence_path, 'mav0/imu0/data.csv')

    # Load data
    image_files, image_timestamps = load_images(image_path, image_timestamps_file)
    imu_timestamps, accelerometer, gyroscope = load_imu(imu_file)

    # Initialize SLAM system
    slam = orbslam3.system(args.vocab_file, args.settings_file, orbslam3.Sensor.IMU_MONOCULAR)
    slam.set_use_viewer(True)
    slam.initialize()
    prev_timestamp = None
    
    # Create processed folder if it doesn't exist
    processed_dir = os.path.join(os.path.dirname(image_files[0]), 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Process sequence
    for img_file, timestamp in zip(image_files, image_timestamps):
        # Read image with better error handling
        try:
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Failed to load image: {img_file}")
                print(f"Checking if file exists: {os.path.exists(img_file)}")
                print(f"File size: {os.path.getsize(img_file) if os.path.exists(img_file) else 'File not found'}")
                continue
                
            # Print image properties for debugging
            print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")
            
        except Exception as e:
            print(f"Error loading image {img_file}: {str(e)}")
            continue

        # Get IMU measurements since last frame
        imu_measurements = []
        if prev_timestamp is not None:
            imu_measurements = get_imu_measurements_between(
                prev_timestamp, timestamp, 
                imu_timestamps, accelerometer, gyroscope
            )

        # Track frame
        tracked = slam.process_image_mono_inertial(img, timestamp, imu_measurements)
        print(f"Timestamp: {timestamp}, Tracking successful: {tracked}")
        
        if "Reseting active map" in str(tracked) or "Reset" in str(tracked):
            print("Map reset detected")
            continue

        # Only try to get trajectory if tracking was successful
        if tracked and isinstance(tracked, bool):  # Make sure it's a boolean True
            try:
                trajectory = slam.get_trajectory()
                if trajectory:  # Only print if we got a valid trajectory
                    print(f"Current trajectory length: {len(trajectory)}")
            except Exception as e:
                print(f"Could not get trajectory: {e}")

        # Move processed image to processed folder
        processed_file = os.path.join(processed_dir, os.path.basename(img_file))
        os.rename(img_file, processed_file)

        prev_timestamp = timestamp
    
    print("Done")
    # Shutdown SLAM system
    # slam.shutdown()
if __name__ == '__main__':
    main()
