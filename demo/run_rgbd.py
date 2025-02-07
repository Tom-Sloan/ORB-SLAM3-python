import time
import orbslam3
import argparse
from glob import glob
import os 
import cv2
import numpy as np
from multiprocessing import shared_memory
import json
import tarfile
import pika
import aio_pika

def main():
    parser = argparse.ArgumentParser(description="Run ORB-SLAM3 Monocular with shared memory")
    parser.add_argument("recording_dir", help="Path to recording directory (e.g., /path/to/20250128_193743)")
    parser.add_argument("--third_party_path", default="./third_party", help="Path to ORB-SLAM3 third party directory")
    args = parser.parse_args()

    # Construct paths based on standard directory structure
    print(args.recording_dir)
    mav0_dir = os.path.join(args.recording_dir, "mav0")
    image_dir = os.path.join(mav0_dir, "cam0", "data")
    settings_file = os.path.join(mav0_dir, "EuRoC_mono.yaml")
    print("settings_file", settings_file)
    # Verify paths exist
    for path in [mav0_dir, image_dir, settings_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")
        
    # Extract ORB vocabulary
    vocab_file = os.path.join(args.third_party_path, "ORB_SLAM3/Vocabulary/ORBvoc.txt")
    if not os.path.exists(vocab_file):
        vocab_file_tar = os.path.join(args.third_party_path, "ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz")
        if not os.path.exists(vocab_file_tar):
            raise FileNotFoundError(f"Required path not found: {vocab_file_tar}")
        else:
            # Extract ORB vocabulary
            with tarfile.open(vocab_file_tar, 'r:gz') as tar:
                tar.extractall(path=args.third_party_path)
                if not os.path.exists(vocab_file):
                    raise FileNotFoundError(f"Failed to extract ORB vocabulary to {args.third_party_path}")
                else:
                    print(f"Extracted ORB vocabulary to {args.third_party_path}")

    # Initialize SLAM system
    slam = orbslam3.system(vocab_file, settings_file, orbslam3.Sensor.MONOCULAR)
    # slam.set_use_viewer(True)
    slam.initialize()

    # Get all PNG files in the rgb/ subdirectory of the dataset path
    # - glob() finds all files matching the pattern 'rgb/*.png'
    # - os.path.join() combines dataset_path with 'rgb/*.png' using proper path separator
    # - sorted() ensures files are in alphabetical/numerical order
    # - Result is a list of full paths to all RGB image files
    img_files = sorted(glob(os.path.join(image_dir, '*.jpg')))

    start_time = time.time()
    frame_count = 0

    processed_files = set()
    last_fps_print = time.time()
    frames_in_window = 0  # Track frames processed in current window

    # Create shared memory for trajectory data
    MAX_POSES = 1000  # Maximum number of poses to store
    # Calculate correct buffer size:
    # - Each pose is 5x4 matrix of float64 (5*4*8 bytes)
    # - Total size = MAX_POSES * (5 * 4 * 8)
    POSE_SIZE = 5 * 4 * 8  # 5x4 matrix of float64
    shm = shared_memory.SharedMemory(create=True, size=MAX_POSES * POSE_SIZE, name='slam_trajectory')
    trajectory_array = np.ndarray((MAX_POSES, 5, 4), dtype=np.float64, buffer=shm.buf)
    trajectory_array.fill(0)  # Initialize with zeros

    # Create shared memory for metadata (current trajectory length and write position)
    shm_meta = shared_memory.SharedMemory(create=True, size=16, name='slam_trajectory_meta')
    meta_array = np.ndarray((2,), dtype=np.int64, buffer=shm_meta.buf)
    meta_array[0] = 0  # number of poses
    meta_array[1] = 0  # write position

    try:
        while True:
            # Get current image files
            current_files = set(glob(os.path.join(image_dir, '*.jpg')))
            
            # Process new files
            new_files = current_files - processed_files
            for img_path in sorted(new_files):
                timestamp = img_path.split('/')[-1][:-4]
                img = cv2.imread(img_path, -1)
                if img is not None:
                    pose = slam.process_image_mono(img, float(timestamp))
                    trajectory = slam.get_trajectory()
                    
                    # Update trajectory in shared memory
                    if trajectory is not None and len(trajectory) > 0:
                        # Get write position
                        write_pos = meta_array[1]
                        
                        # Write new pose
                        trajectory_array[write_pos, :4, :] = trajectory[-1]  # Store latest pose
                        trajectory_array[write_pos, 4, 0] = float(timestamp)
                        
                        # Update write position
                        write_pos = (write_pos + 1) % MAX_POSES
                        meta_array[1] = write_pos
                        
                        # Update number of poses
                        meta_array[0] = min(meta_array[0] + 1, MAX_POSES)
                    
                    frame_count += 1
                    frames_in_window += 1
                    processed_files.add(img_path)

            # Print FPS stats every 10 seconds if new frames were processed
            current_time = time.time()
            if current_time - last_fps_print >= 10 and frames_in_window > 0:
                window_time = current_time - last_fps_print
                fps = frames_in_window / window_time
                print(f"\nProcessed {frames_in_window} frames in last {window_time:.2f} seconds")
                print(f"Current FPS: {fps:.2f}")
                last_fps_print = current_time
                frames_in_window = 0  # Reset counter for next window

            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)

    except Exception as e:
        print(f"Error occurred: {e}")
        try:
            shm.close()
            shm_meta.close()
            # shm.unlink()
            # shm_meta.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        # Clean up shared memory
        print("Cleaning up shared memory...")
        try:
            shm.close()
            shm_meta.close()
            # shm.unlink()
            # shm_meta.unlink()
        except FileNotFoundError:
            pass

async def process_images():
    connection = await aio_pika.connect(os.getenv('RABBITMQ_URL'))
    channel = await connection.channel()
    queue = await channel.declare_queue('image_data')
    
    async for message in queue:
        async with message.process():
            data = json.loads(message.body)
            # Process frame with SLAM
            pose = slam.process_image_mono(cv2.imdecode(data['frame']))
            # Publish pose to trajectory queue

class SLAMProcessor:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.URLParameters(os.getenv('RABBITMQ_URL')))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='image_data')
        self.channel.queue_declare(queue='imu_data')
        
    def process_messages(self):
        def callback(ch, method, properties, body):
            data = json.loads(body)
            # Process frame with SLAM
            pose = self.slam.process_image_mono(data['frame'], data['timestamp'])
            # Publish trajectory updates
            ch.basic_publish(
                exchange='',
                routing_key='trajectory_updates',
                body=json.dumps(pose.tolist()))
            
        self.channel.basic_consume(
            queue='image_data',
            on_message_callback=callback,
            auto_ack=True)
        self.channel.start_consuming()

if __name__ == "__main__":
    main()