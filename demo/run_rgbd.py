#!/usr/bin/env python3
"""
run_rgbd.py
-----------
Continuously listens for images (and optionally IMU data) from RabbitMQ queues, 
runs ORB-SLAM3 in either monocular or mono-inertial mode, 
and publishes resulting trajectory poses to a fanout exchange 'trajectory_exchange'.

Requires:
  pip install pika opencv-python-headless numpy orbslam3 (plus your other dependencies)

Environment:
  RABBITMQ_URL=amqp://rabbitmq  # or similar
  SLAM_MODE=mono or mono_inertial
  (or pass '--mode mono'/'--mode mono_inertial' as a command-line arg)
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pika
import argparse
import base64

# If you have orbslam3 installed as a Python module:
import orbslam3


def load_slam_system(slam_mode: str):
    """
    Initialize ORB-SLAM3 in the chosen mode ('mono' or 'mono_inertial')
    using local config files in the same folder as run_rgbd.py.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Adjust these as needed or rename them in your project
    mono_config = os.path.join(this_dir, "EuRoC_mono.yaml")
    inertial_config = os.path.join(this_dir, "EuRoC_mono_inertial.yaml")

    # Path to ORB vocabulary
    # (In real usage, ensure your ORBvoc.txt or .bin is here)
    vocabulary_path = os.path.join('third_party/ORB_SLAM3/Vocabulary/', "ORBvoc.txt")
    if not os.path.exists(vocabulary_path):
        # If it's not in the same folder, adapt accordingly
        raise FileNotFoundError(f"ORB vocabulary file not found: {vocabulary_path}")

    if slam_mode == "mono":
        slam = orbslam3.system(
            vocabulary_path,
            mono_config,
            orbslam3.Sensor.MONOCULAR
        )
    elif slam_mode == "mono_inertial":
        slam = orbslam3.system(
            vocabulary_path,
            inertial_config,
            orbslam3.Sensor.IMU_MONOCULAR
        )
    else:
        raise ValueError(f"Unknown SLAM mode: {slam_mode}")

    slam.initialize()
    return slam


class RunRGBD:
    """
    Example SLAM processor that:
     - Subscribes to image_data_exchange (and optional IMU)
     - Runs ORB-SLAM3 in either monocular or mono-inertial mode
     - Publishes resulting trajectory to a new fanout exchange ('trajectory_exchange').
    """

    def __init__(self, slam_mode="mono"):
        self.slam_mode = slam_mode

        # We store images + IMU in memory if we need to unify them.
        # For a real solution, you'd want a more robust approach (timestamp matching, etc.).
        self.imu_buffer = []
        self.slam = load_slam_system(self.slam_mode)

        # ---- Setup RabbitMQ Connection ----
        amqp_url = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
        params = pika.URLParameters(amqp_url)
        params.heartbeat = 3600  # 1-hour heartbeat
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()

        # We will create unique queues for this consumer and bind them to the fanout exchanges
        # that 'frame_processor.py' is publishing to.
        # image_data_exchange -> image_data_slam queue
        # imu_data_exchange   -> imu_data_slam queue  (only if we are in mono_inertial mode)

        # 1) Declare fanout exchange (already done by publisher, but safe to re-declare):
        self.channel.exchange_declare(
            exchange='image_data_exchange',
            exchange_type='fanout',
            durable=True
        )
        self.channel.exchange_declare(
            exchange='imu_data_exchange',
            exchange_type='fanout',
            durable=True
        )

        # 2) Declare a queue for images
        res_image = self.channel.queue_declare(queue='', exclusive=True)  # let Rabbit pick name
        self.image_queue = res_image.method.queue
        # Bind it
        self.channel.queue_bind(
            exchange='image_data_exchange',
            queue=self.image_queue,
            routing_key=''
        )

        self.imu_queue = None
        if self.slam_mode == "mono_inertial":
            res_imu = self.channel.queue_declare(queue='', exclusive=True)  # ephemeral queue
            self.imu_queue = res_imu.method.queue
            # Bind it
            self.channel.queue_bind(
                exchange='imu_data_exchange',
                queue=self.imu_queue,
                routing_key=''
            )

        # 3) Declare a new fanout exchange to publish trajectory updates
        # (multiple subscribers can read from it)
        self.channel.exchange_declare(
            exchange='trajectory_data_exchange',
            exchange_type='fanout',
            durable=True
        )

        # 4) Set up consumers
        self.channel.basic_consume(
            queue=self.image_queue,
            on_message_callback=self.on_image_message,
            auto_ack=True
        )

        if self.imu_queue:
            self.channel.basic_consume(
                queue=self.imu_queue,
                on_message_callback=self.on_imu_message,
                auto_ack=True
            )

        print(f" [*] Subscribed to fanout exchange(s) in mode={self.slam_mode}")
        print(f"     Image queue: {self.image_queue}")
        if self.imu_queue:
            print(f"     IMU queue:   {self.imu_queue}")

    def on_image_message(self, ch, method, properties, body):
        """
        Callback for image messages from 'image_data_exchange'.
        Each message includes 'frame_data' (base64 image) and 'timestamp'.
        """
        try:
            data = json.loads(body)
            frame_b64 = data['frame_data']
            frame_ts_ns = data['timestamp']  # e.g. 1234567890123456

            # Convert base64 -> image
            frame_bytes = cv2.imdecode(
                np.frombuffer(
                    base64.b64decode(frame_b64),
                    np.uint8
                ),
                cv2.IMREAD_COLOR
            )
            if frame_bytes is None:
                print(" [!] Received invalid image frame.")
                return

            # Convert timestamp from ns to seconds
            frame_ts_s = frame_ts_ns / 1e9

            # For a real solution, you'd gather all IMU up to 'frame_ts_s'
            # for mono_inertial. We'll do a naive approach, ignoring partial sync.
            if self.slam_mode == "mono_inertial":
                # We may want to collect IMU from self.imu_buffer that is <= frame_ts_s
                # to pass into slam
                # For demonstration, let's pass all
                imu_measurements = self.imu_buffer[:]
                self.imu_buffer.clear()

                # Call ORB-SLAM
                success = self.slam.process_image_mono_inertial(frame_bytes, frame_ts_s, imu_measurements)
            else:
                # Monocular only
                success = self.slam.process_image_mono(frame_bytes, frame_ts_s)

            # Extract the current pose from get_trajectory() or from the returned pose
            current_pose = None
            if success:
                # Some versions of orbslam3 Python wrapper return a matrix, or you might do:
                current_pose = self.slam.get_trajectory()[-1]
            else:
                # or you might call self.slam.get_trajectory(), etc.
                pass
            
            last_pose = self.slam.get_trajectory()  # returns the entire list of poses
            if last_pose and len(last_pose) > 0:
                # Grab the last pose only
                pose = last_pose[-1]
                if isinstance(pose, np.ndarray):
                    pose_list = pose.tolist()
                else:
                    pose_list = pose

                msg = {
                    "timestamp_ns": frame_ts_ns,  # or you can store the current time
                    "pose": pose_list
                }
                self.channel.basic_publish(
                    exchange='trajectory_data_exchange',
                    routing_key='',
                    body=json.dumps(msg)
                )
                print(f"Published LATEST pose at {frame_ts_ns} ns")

        except Exception as e:
            print(f"Error in on_image_message: {e}")

    def on_imu_message(self, ch, method, properties, body):
        """
        Callback for IMU messages from 'imu_data_exchange'.
        data includes 'timestamp' (ns), 'angular_velocity', 'linear_acceleration'
        """
        try:
            data = json.loads(body)
            imu_ts_ns = data['timestamp']
            gyro = data['angular_velocity']   # [gx, gy, gz]
            accel = data['linear_acceleration']  # [ax, ay, az]

            # Convert to orbslam3.IMU.Point if your orbslam3 binding requires that
            # We'll store in a simple dict for demonstration
            imu_point = {
                "timestamp_s": imu_ts_ns / 1e9,
                "gyro": gyro,
                "accel": accel
            }
            # Append to buffer
            self.imu_buffer.append(imu_point)

        except Exception as e:
            print(f"Error in on_imu_message: {e}")

    def run_forever(self):
        """
        Start consuming from queues forever. This call is blocking.
        """
        print(" [*] Starting blocking consume. Press Ctrl+C to stop.")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("\n [x] Interrupted by user")

        # Clean up
        self.slam.shutdown()
        self.connection.close()


if __name__ == "__main__":
    # Allow user to pass --mode mono or --mode mono_inertial
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=os.getenv("SLAM_MODE", "mono"),
                        choices=["mono", "mono_inertial"],
                        help="SLAM mode to run: 'mono' or 'mono_inertial'")
    args = parser.parse_args()

    node = RunRGBD(slam_mode=args.mode)
    node.run_forever()
