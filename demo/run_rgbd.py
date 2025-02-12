#!/usr/bin/env python3
"""
run_rgbd.py
-----------
Continuously listens for images (and optionally IMU data) from RabbitMQ queues, 
runs ORB-SLAM3 in either monocular or mono-inertial mode, 
and publishes resulting trajectory poses to a fanout exchange 'trajectory_exchange'
as well as a combined message to a new exchange 'slam_data_exchange'.
...
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

# PROMETHEUS
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Define metrics
slam_frames_processed = Counter(
    "slam_frames_processed_total",
    "Number of frames processed by the SLAM system"
)

slam_frame_latency = Histogram(
    "slam_frame_processing_seconds",
    "Time spent processing each frame in SLAM",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]
)

slam_fps_gauge = Gauge(
    "slam_frames_per_second",
    "Frames per second of the SLAM pipeline"
)

# Get exchange names from the environment
VIDEO_FRAMES_EXCHANGE = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
IMU_DATA_EXCHANGE = os.getenv("IMU_DATA_EXCHANGE", "imu_data_exchange")
TRAJECTORY_DATA_EXCHANGE = os.getenv("TRAJECTORY_DATA_EXCHANGE", "trajectory_data_exchange")
RESTART_EXCHANGE = os.getenv("RESTART_EXCHANGE", "restart_exchange")
SLAM_DATA_EXCHANGE = os.getenv("SLAM_DATA_EXCHANGE", "slam_data_exchange")  # New exchange

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
    vocabulary_path = os.path.join('third_party/ORB_SLAM3/Vocabulary/', "ORBvoc.txt")
    if not os.path.exists(vocabulary_path):
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
     - Publishes resulting trajectory to a new fanout exchange ('trajectory_exchange')
       and publishes a combined SLAM message (image+pose) to 'slam_data_exchange'.
    """

    def __init__(self, slam_mode="mono"):
        self.slam_mode = slam_mode
        self.failed_tracking_counter = 0
        self.restart_sent = False

        # We store images + IMU in memory if we need to unify them.
        self.imu_buffer = []
        self.slam = load_slam_system(self.slam_mode)

        # ---- Setup RabbitMQ Connection ----
        amqp_url = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
        params = pika.URLParameters(amqp_url)
        params.heartbeat = 3600  # 1-hour heartbeat
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()

        # Declare fanout exchanges:
        self.channel.exchange_declare(
            exchange=VIDEO_FRAMES_EXCHANGE,
            exchange_type='fanout',
            durable=True
        )
        self.channel.exchange_declare(
            exchange=IMU_DATA_EXCHANGE,
            exchange_type='fanout',
            durable=True
        )
        self.channel.exchange_declare(
            exchange=TRAJECTORY_DATA_EXCHANGE,
            exchange_type='fanout',
            durable=True
        )
        # Declare the new combined exchange
        self.channel.exchange_declare(
            exchange=SLAM_DATA_EXCHANGE,
            exchange_type='fanout',
            durable=True
        )

        # 2) Declare a queue for images
        res_image = self.channel.queue_declare(queue='slam_video_input', exclusive=True)
        self.image_queue = res_image.method.queue
        self.channel.queue_bind(
            exchange=VIDEO_FRAMES_EXCHANGE,
            queue=self.image_queue,
            routing_key=''
        )

        self.imu_queue = None
        if self.slam_mode == "mono_inertial":
            res_imu = self.channel.queue_declare(queue='slam_imu_input', exclusive=True)
            self.imu_queue = res_imu.method.queue
            self.channel.queue_bind(
                exchange=IMU_DATA_EXCHANGE,
                queue=self.imu_queue,
                routing_key=''
            )

        self.channel.exchange_declare(
            exchange=TRAJECTORY_DATA_EXCHANGE,
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
            
        self.channel.exchange_declare(
            exchange=RESTART_EXCHANGE,
            exchange_type='fanout',
            durable=True
        )
        res_restart = self.channel.queue_declare(queue='restart_slam', durable=True)
        self.channel.queue_bind(
            exchange=RESTART_EXCHANGE,
            queue=res_restart.method.queue,
            routing_key=''
        )
        self.channel.basic_consume(
            queue=res_restart.method.queue,
            on_message_callback=self._handle_restart,
            auto_ack=True
        )

        print(f" [*] Subscribed to fanout exchange(s) in mode={self.slam_mode}")
        print(f"     Image queue: {self.image_queue}")
        if self.imu_queue:
            print(f"     IMU queue:   {self.imu_queue}")

    def on_image_message(self, ch, method, properties, body):
        start_time = time.time()
        try:
            # Ensure the header contains "timestamp_ns"; if not, discard the message.
            if not (properties and properties.headers and "timestamp_ns" in properties.headers):
                print("[!] No timestamp in image message; discarding message.")
                return
            frame_ts_ns = int(properties.headers.get("timestamp_ns"))
            # Decode the raw JPEG bytes directly
            frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print(" [!] Received invalid image frame.")
                return

            frame_ts_s = frame_ts_ns / 1e9

            if self.slam_mode == "mono_inertial":
                imu_measurements = self.imu_buffer[:]
                self.imu_buffer.clear()
                success = self.slam.process_image_mono_inertial(frame, frame_ts_s, imu_measurements)
            else:
                success = self.slam.process_image_mono(frame, frame_ts_s)

            slam_frames_processed.inc()
            elapsed = time.time() - start_time
            slam_frame_latency.observe(elapsed)
            slam_fps_gauge.set(1.0 / elapsed if elapsed > 0 else 0.0)

            # --- (1) Always publish the current trajectory if available ---
            last_pose = self.slam.get_trajectory()
            if last_pose and len(last_pose) > 0:
                pose = last_pose[-1]
                # Convert to list if needed (for JSON serialization)
                pose_list = pose.tolist() if isinstance(pose, np.ndarray) else pose
                # Publish to the original TRAJECTORY_DATA_EXCHANGE
                msg = {
                    "timestamp_ns": frame_ts_ns,
                    "pose": pose_list,
                    "tracking_success": success
                }
                self.channel.basic_publish(
                    exchange=TRAJECTORY_DATA_EXCHANGE,
                    routing_key='',
                    body=json.dumps(msg)
                )
                print(f"Published LATEST pose at {frame_ts_ns} ns, tracking: {success}")

                # --- (2) Also publish a combined SLAM message (image + trajectory) ---
                combined_msg = {
                    "timestamp_ns": frame_ts_ns,
                    "pose": pose_list,
                    "tracking_success": success,
                    # Base64 encode the original image bytes (JPEG)
                    "image_b64": base64.b64encode(body).decode('utf-8')
                }
                self.channel.basic_publish(
                    exchange=SLAM_DATA_EXCHANGE,
                    routing_key='',
                    body=json.dumps(combined_msg)
                )
                print(f"Published combined SLAM data at {frame_ts_ns} ns")

            # --- (3) Detect tracking failure and publish restart if needed ---
            if not success:
                self.failed_tracking_counter += 1
                # For example, if there is a failure (or consecutive failures), trigger a restart.
                if self.failed_tracking_counter >= 1 and not self.restart_sent:
                    print("Tracking failure threshold reached, sending restart command")
                    restart_msg = json.dumps({"type": "restart"})
                    self.channel.basic_publish(
                        exchange=RESTART_EXCHANGE,
                        routing_key='',
                        body=restart_msg
                    )
                    self.restart_sent = True
                    self.failed_tracking_counter = 0  # reset the counter after issuing restart
            else:
                # Reset failure count and flag on success.
                self.failed_tracking_counter = 0
                self.restart_sent = False

        except Exception as e:
            print(f"Error in on_image_message: {e}")

    def on_imu_message(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            if 'timestamp' not in data or 'imu_data' not in data:
                print("[!] IMU message missing 'timestamp' or 'imu_data'; discarding message.")
                return
            imu_ts = data['timestamp']
            imu_data = data['imu_data']
            if 'velocity' not in imu_data or 'attitude' not in imu_data:
                print("[!] IMU message missing 'velocity' or 'attitude' in 'imu_data'; discarding message.")
                return
            velocity = imu_data['velocity']
            attitude = imu_data['attitude']
            imu_point = {
                "timestamp_s": imu_ts / 1e9,
                "gyro": [attitude.get('pitch', 0), attitude.get('roll', 0), attitude.get('yaw', 0)],
                "accel": [velocity.get('x', 0), velocity.get('y', 0), velocity.get('z', 0)]
            }
            self.imu_buffer.append(imu_point)
        except Exception as e:
            print(f"Error in on_imu_message: {e}")

    def _handle_restart(self, ch, method, properties, body):
        try:
            msg = json.loads(body)
            if msg.get("type") == "restart":
                print("Restart command received in SLAM. Refreshing SLAM system...")
                self.slam.shutdown()
                self.slam = load_slam_system(self.slam_mode)
                self.imu_buffer.clear()
        except Exception as e:
            print("Error handling restart in SLAM:", e)

    def run_forever(self):
        print(" [*] Starting blocking consume. Press Ctrl+C to stop.")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("\n [x] Interrupted by user")
        self.slam.shutdown()
        self.connection.close()


if __name__ == "__main__":
    start_http_server(8000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=os.getenv("SLAM_MODE", "mono"),
                        choices=["mono", "mono_inertial"],
                        help="SLAM mode to run: 'mono' or 'mono_inertial'")
    args = parser.parse_args()

    node = RunRGBD(slam_mode=args.mode)
    node.run_forever()
