import time
import orbslam3
import argparse
from glob import glob
import os
import cv2
import numpy as np
import csv
from multiprocessing import shared_memory
import json
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import threading

from orbslam3 import IMU

def load_imu(imu_path):
    """
    Load IMU measurements (timestamp [ns], gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z)
    Returns:
        imu_timestamps (list of floats in seconds),
        accelerometer (list of [ax, ay, az]),
        gyroscope (list of [gx, gy, gz])
    """
    imu_timestamps = []
    accelerometer = []
    gyroscope = []
    
    with open(imu_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            # Convert from ns to s
            timestamp_s = float(row[0]) / 1e9
            gyro = [float(row[1]), float(row[2]), float(row[3])]
            acc = [float(row[4]), float(row[5]), float(row[6])]
            
            imu_timestamps.append(timestamp_s)
            gyroscope.append(gyro)
            accelerometer.append(acc)
            
    return imu_timestamps, accelerometer, gyroscope

def get_imu_measurements_between(t0, t1, imu_timestamps, accelerometer, gyroscope, last_imu_index):
    """
    Collect and return all IMU measurements between two timestamps (exclusive of t0, inclusive of t1).
    Returns a list of orbslam3.IMU.Point and the updated last_imu_index to continue from.
    """
    measurements = []
    start_index = last_imu_index
    while start_index < len(imu_timestamps) and imu_timestamps[start_index] <= t0:
        start_index += 1
    
    new_imu_index = start_index
    while new_imu_index < len(imu_timestamps) and imu_timestamps[new_imu_index] <= t1:
        ts = imu_timestamps[new_imu_index]
        acc_vals = accelerometer[new_imu_index]
        gyro_vals = gyroscope[new_imu_index]
        point = IMU.Point(acc_vals[0], acc_vals[1], acc_vals[2],
                          gyro_vals[0], gyro_vals[1], gyro_vals[2],
                          ts)
        measurements.append(point)
        new_imu_index += 1
    
    # Return the new index so we don't re-scan IMU data next time
    return measurements, new_imu_index

class NewRecordingHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.dir_pattern = re.compile(r'^\d{8}_\d{6}$')  # YYYYMMDD_HHMMSS format
    
    def on_created(self, event):
        if event.is_directory and self.dir_pattern.match(os.path.basename(event.src_path)):
            print(f"Detected new directory: {event.src_path}")
            self.callback()

def find_latest_recording(parent_dir):
    """Find the most recent valid recording directory"""
    dirs = [d for d in os.listdir(parent_dir) 
           if os.path.isdir(os.path.join(parent_dir, d)) 
           and re.match(r'^\d{8}_\d{6}$', d)]
    
    if not dirs:
        return None
        
    sorted_dirs = sorted(dirs, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S"), reverse=True)
    return os.path.join(parent_dir, sorted_dirs[0])

def process_recording(recording_dir, args, stop_event):
    """Process a single recording directory"""
    try:
        mav0_dir = os.path.join(recording_dir, "mav0")
        image_dir = os.path.join(mav0_dir, "cam0", "data")
        imu_dir = os.path.join(mav0_dir, "imu0", "data")
        settings_file = os.path.join(mav0_dir, "EuRoC_mono_inertial.yaml")

        # Verify paths exist
        for path in [mav0_dir, image_dir, imu_dir, settings_file]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path not found: {path}")

        # Initialize SLAM system
        slam = orbslam3.system(args.vocab_file, settings_file, orbslam3.Sensor.IMU_MONOCULAR)
        slam.set_use_viewer(True)
        slam.initialize()

        # Get first image to establish time origin
        img_files = sorted(glob(os.path.join(image_dir, '*.jpg')))
        if not img_files:
            raise FileNotFoundError(f"No images found in {image_dir}")
        
        first_image = img_files[0]
        first_timestamp_ns = int(os.path.splitext(os.path.basename(first_image))[0])
        print(f"First timestamp (ns): {first_timestamp_ns}")

        # Shared memory for trajectory data (same structure as run_rgbd.py)
        MAX_POSES = 1000
        # Each pose is 5x4 of float64 => 5*4*8 bytes
        POSE_SIZE = 5 * 4 * 8
        try:
            # Create or attach shared memory
            shm = shared_memory.SharedMemory(create=True, size=MAX_POSES * POSE_SIZE, name='slam_trajectory')
        except FileExistsError:
            # In case it was not cleaned up properly before
            shm = shared_memory.SharedMemory(create=False, name='slam_trajectory')
        trajectory_array = np.ndarray((MAX_POSES, 5, 4), dtype=np.float64, buffer=shm.buf)
        trajectory_array.fill(0)

        # Shared memory for metadata
        try:
            shm_meta = shared_memory.SharedMemory(create=True, size=16, name='slam_trajectory_meta')
        except FileExistsError:
            shm_meta = shared_memory.SharedMemory(create=False, name='slam_trajectory_meta')
        meta_array = np.ndarray((2,), dtype=np.int64, buffer=shm_meta.buf)
        meta_array[0] = 0  # number of poses
        meta_array[1] = 0  # write position

        processed_files = set()
        processed_imu_files = set()  # Track processed IMU files
        frame_count = 0
        frames_in_window = 0
        last_fps_print = time.time()
        prev_timestamp_ns = None

        print("Entering main loop. Press Ctrl+C to exit.")

        while not stop_event.is_set():
            current_files = set(glob(os.path.join(image_dir, '*.jpg')))
            new_files = sorted(current_files - processed_files, 
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            for img_path in new_files:
                # Extract timestamp from filename as integer
                try:
                    basename = os.path.basename(img_path)
                    timestamp_ns = int(os.path.splitext(basename)[0])
                except ValueError:
                    print(f"Skipping invalid filename: {basename}")
                    continue
                
                # Calculate relative timestamp in seconds with nanosecond precision
                rel_timestamp = (timestamp_ns - first_timestamp_ns) / 1e9
                
                # Load image with verification
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    print(f"Failed to load image: {img_path}")
                    processed_files.add(img_path)
                    continue

                # Collect IMU measurements
                imu_measurements = []
                if prev_timestamp_ns is not None:
                    # Find IMU files between previous and current image
                    imu_files = sorted(
                        glob(os.path.join(imu_dir, '*.txt')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                    )
                    
                    for imu_path in imu_files:
                        if imu_path in processed_imu_files:
                            continue
                            
                        # Extract IMU timestamp from filename
                        imu_basename = os.path.basename(imu_path)
                        imu_ts_ns = int(os.path.splitext(imu_basename)[0])
                        
                        # Only process if between frames and not already processed
                        if prev_timestamp_ns < imu_ts_ns <= timestamp_ns:
                            try:
                                with open(imu_path, 'r') as f:
                                    line = f.readline().strip()
                                    values = list(map(float, line.split()))
                                    if len(values) != 6:
                                        print(f"Invalid IMU data in {imu_basename}")
                                        continue
                                    
                                    # Create IMU point with relative timestamp
                                    imu_rel_ts = (imu_ts_ns - first_timestamp_ns) / 1e9
                                    point = IMU.Point(
                                        values[3], values[4], values[5],  # acc
                                        values[0], values[1], values[2],  # gyro
                                        imu_rel_ts
                                    )
                                    imu_measurements.append(point)
                                    processed_imu_files.add(imu_path)
                            except Exception as e:
                                print(f"Error processing {imu_basename}: {str(e)}")
                                continue

                # Print debug info
                print(f"Processing {basename} @ {rel_timestamp:.6f}s with {len(imu_measurements)} IMU readings")

                # Process frame
                pose = slam.process_image_mono_inertial(img, rel_timestamp, imu_measurements)

                # Fetch trajectory if process was successful
                # Note: Usually, ORB-SLAM3 returns the current pose or success/fail status
                # but let's check the trajectory if not None
                trajectory = slam.get_trajectory()
                if trajectory is not None and len(trajectory) > 0:
                    # Write the latest pose to shared memory
                    write_pos = meta_array[1]
                    trajectory_array[write_pos, :4, :] = trajectory[-1]
                    trajectory_array[write_pos, 4, 0] = rel_timestamp
                    # Fill rest of row with zeros just in case
                    trajectory_array[write_pos, 4, 1:] = 0.0

                    write_pos = (write_pos + 1) % MAX_POSES
                    meta_array[1] = write_pos
                    meta_array[0] = min(meta_array[0] + 1, MAX_POSES)

                processed_files.add(img_path)
                frame_count += 1
                frames_in_window += 1
                prev_timestamp_ns = timestamp_ns

            # Print FPS stats every 10 seconds if new frames were processed
            current_time = time.time()
            if current_time - last_fps_print >= 10 and frames_in_window > 0:
                window_time = current_time - last_fps_print
                fps = frames_in_window / window_time
                print(f"\nProcessed {frames_in_window} frames in last {window_time:.2f} seconds")
                print(f"Current FPS: {fps:.2f}")
                last_fps_print = current_time
                frames_in_window = 0

            elif current_time - last_fps_print >= 10 and frames_in_window == 0:
                print("No new frames in last 10 seconds.")

            # Sleep to prevent high CPU usage
            time.sleep(0.001)

    finally:
        if 'slam' in locals():
            slam.shutdown()
        print(f"Stopped processing {recording_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings_root", help="Path to parent directory of recordings")
    parser.add_argument("--vocab_file", required=True)
    args = parser.parse_args()

    current_process = None
    stop_event = threading.Event()
    observer = Observer()

    def start_new_processing():
        nonlocal current_process, stop_event
        stop_event.set()
        if current_process and current_process.is_alive():
            current_process.join()
        
        latest = find_latest_recording(args.recordings_root)
        if latest:
            stop_event = threading.Event()
            current_process = threading.Thread(
                target=process_recording, 
                args=(latest, args, stop_event)
            )
            current_process.start()

    # Set up directory watcher
    event_handler = NewRecordingHandler(start_new_processing)
    observer.schedule(event_handler, args.recordings_root, recursive=False)
    observer.start()

    try:
        # Initial start
        start_new_processing()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        observer.stop()
    finally:
        observer.join()
        if current_process and current_process.is_alive():
            current_process.join()

if __name__ == "__main__":
    main()
