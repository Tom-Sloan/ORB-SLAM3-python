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
import yaml
import random
import threading
from collections import deque

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

slam_reset_counter = Counter(
    "slam_reset_total",
    "Number of times the SLAM system has been reset"
)

slam_tracking_success_rate = Gauge(
    "slam_tracking_success_rate",
    "Percentage of frames with successful tracking in the last minute"
)

# Get exchange names from the environment
VIDEO_FRAMES_EXCHANGE = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
IMU_DATA_EXCHANGE = os.getenv("IMU_DATA_EXCHANGE", "imu_data_exchange")
TRAJECTORY_DATA_EXCHANGE = os.getenv("TRAJECTORY_DATA_EXCHANGE", "trajectory_data_exchange")
RESTART_EXCHANGE = os.getenv("RESTART_EXCHANGE", "restart_exchange")
SLAM_DATA_EXCHANGE = os.getenv("SLAM_DATA_EXCHANGE", "slam_data_exchange")  
PROCESSED_IMU_EXCHANGE = os.getenv("PROCESSED_IMU_EXCHANGE", "processed_imu_exchange")  # New exchange for processed IMU data

DRONE_CONFIG_PATH = os.environ.get("DRONE_CONFIG_PATH", "/app/drone_config.yaml")

def load_drone_config(path):
    # Just check if the file exists, no need to parse correctly
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # We can read the file to check basic validity, but don't need to parse the opencv matrix
    with open(path, 'r') as f:
        # Return an empty dict to indicate the file exists
        # The C++ ORB-SLAM3 will parse the file properly
        return {}

def load_slam_system(slam_mode: str):
    """
    Initialize ORB-SLAM3 in the chosen mode ('mono' or 'mono_inertial')
    using local config files in the same folder as run_rgbd.py.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        drone_cfg = load_drone_config(DRONE_CONFIG_PATH)
        print(f"[continuous_reconstruction.py] Loaded drone_config.yaml from: {DRONE_CONFIG_PATH}")

    except FileNotFoundError:
        print(f"[ERROR] Could not find drone_config.yaml at {DRONE_CONFIG_PATH}")
        raise FileNotFoundError(f"[ERROR] Could not find drone_config.yaml at {DRONE_CONFIG_PATH}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load drone_config.yaml: {e}")
        raise Exception(f"[ERROR] Failed to load drone_config.yaml: {e}")

    # Adjust these as needed or rename them in your project
    mono_config = os.path.join(this_dir, "EuRoC_mono.yaml")
    inertial_config = DRONE_CONFIG_PATH

    # Path to ORB vocabulary
    vocabulary_path = os.path.join('third_party/ORB_SLAM3/Vocabulary/', "ORBvoc.txt")
    if not os.path.exists(vocabulary_path):
        raise FileNotFoundError(f"ORB vocabulary file not found: {vocabulary_path}")
    if slam_mode == "mono":
        print(f"[continuous_reconstruction.py] Loaded monocular_config.yaml from: {mono_config}")
        slam = orbslam3.system(
            vocabulary_path,
            mono_config,
            orbslam3.Sensor.MONOCULAR
        )
    elif slam_mode == "mono_inertial":
        print(f"[continuous_reconstruction.py] Loaded inertial_config.yaml from: {inertial_config}")
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

    def __init__(self, slam_mode="mono_inertial"):
        self.slam_mode = slam_mode
        self.failed_tracking_counter = 0
        self.restart_sent = False
        self.imu_initialized = False

        # Reset transformation for trajectory continuity
        self.reset_transform = None
        self.last_valid_transformation = None
        self.reset_count = 0
        
        # Tracking success metric
        self.tracking_history = deque(maxlen=300)  # 300 frames ~= 10 seconds @ 30fps
        
        # Frame buffering parameters
        self.MAX_FRAME_BUFFER_SIZE = 20  # Maximum frames to buffer
        self.MIN_IMU_COUNT = 3           # Minimum IMU measurements required
        self.IMU_WAIT_TIMEOUT = 1.0      # Max time to wait for IMU data (seconds)
        self.RESET_THRESHOLD = 10        # Number of consecutive tracking failures before reset
        
        # Buffers
        self.imu_buffer = []             # IMU buffer
        self.frame_buffer = []           # New buffer for frames waiting for IMU
        self.last_image_timestamps = []  # Timestamps buffer
        self.imu_moving_avg = []         # For IMU smoothing
        
        # Process control
        self.processing_lock = threading.Lock()
        
        # Load SLAM system
        self.slam = load_slam_system(self.slam_mode)

        # ---- Setup RabbitMQ Connection ----
        self._setup_rabbitmq()
        
        # Start buffer checking thread
        self.buffer_check_thread = threading.Thread(target=self._check_frame_buffer)
        self.buffer_check_thread.daemon = True
        self.buffer_check_thread.start()

    def _setup_rabbitmq(self):
        """Setup RabbitMQ connection and channels"""
        try:
            amqp_url = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
            params = pika.URLParameters(amqp_url)
            params.heartbeat = 3600  # 1-hour heartbeat
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()

            # Declare fanout exchanges
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
            self.channel.exchange_declare(
                exchange=SLAM_DATA_EXCHANGE,
                exchange_type='fanout',
                durable=True
            )
            self.channel.exchange_declare(
                exchange=PROCESSED_IMU_EXCHANGE,
                exchange_type='fanout',
                durable=True
            )
            self.channel.exchange_declare(
                exchange=RESTART_EXCHANGE,
                exchange_type='fanout',
                durable=True
            )

            # Declare queues and bindings
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

            res_restart = self.channel.queue_declare(queue='restart_slam', durable=True)
            self.channel.queue_bind(
                exchange=RESTART_EXCHANGE,
                queue=res_restart.method.queue,
                routing_key=''
            )

            # Set up consumers
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

            self.channel.basic_consume(
                queue=res_restart.method.queue,
                on_message_callback=self._handle_restart,
                auto_ack=True
            )

            print(f" [*] Subscribed to fanout exchange(s) in mode={self.slam_mode}")
            print(f"     Image queue: {self.image_queue}")
            if self.imu_queue:
                print(f"     IMU queue:   {self.imu_queue}")
        
        except Exception as e:
            print(f"Error setting up RabbitMQ: {e}")
            import traceback
            traceback.print_exc()

    def _check_frame_buffer(self):
        """Periodically checks the frame buffer for frames that can be processed."""
        while True:
            try:
                time.sleep(0.05)  # Check every 50ms
                
                with self.processing_lock:
                    if not self.frame_buffer:
                        continue
                    
                    # Check frames in buffer, oldest first
                    i = 0
                    while i < len(self.frame_buffer):
                        frame_info = self.frame_buffer[i]
                        frame = frame_info["frame"]
                        body = frame_info["body"]
                        frame_ts_s = frame_info["timestamp"]
                        properties = frame_info["properties"]
                        
                        # Count available IMU measurements for this frame
                        imu_count = self._count_imu_for_frame(frame_ts_s)
                        
                        if imu_count >= self.MIN_IMU_COUNT:
                            # Sufficient IMU data available, remove from buffer and process
                            self.frame_buffer.pop(i)
                            # Create a copy of the data before releasing the lock
                            frame_copy = frame.copy()
                            body_copy = body
                            
                            # Release lock before processing
                            self.processing_lock.release()
                            try:
                                # Process in the current thread to avoid threading issues
                                self._process_frame(frame_copy, body_copy, frame_ts_s, properties)
                            finally:
                                # Re-acquire lock
                                self.processing_lock.acquire()
                        elif time.time() - frame_info["buffer_time"] > self.IMU_WAIT_TIMEOUT:
                            # Timeout waiting for IMU data, drop the frame
                            print(f"Timeout waiting for IMU data for frame at {frame_ts_s}, dropping frame")
                            self.frame_buffer.pop(i)
                        else:
                            # Keep waiting for more IMU data
                            i += 1
            except Exception as e:
                print(f"Error in _check_frame_buffer: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Wait a bit before trying again

    def _count_imu_for_frame(self, frame_ts_s):
        """Count available IMU measurements for a frame timestamp."""
        count = 0
        for imu in self.imu_buffer:
            if imu["timestamp_s"] <= frame_ts_s:
                count += 1
        return count
    
    def validate_imu_values(self, gyro, accel):
        """Enhanced validation for IMU values with improved stability checks"""
        # Adjust these thresholds based on your hardware and motion patterns
        MAX_GYRO = 5.0  # rad/s - maximum reasonable gyroscope reading
        MAX_ACCEL = 20.0  # m/s² - maximum reasonable accelerometer reading
        MIN_ACCEL = 0.5  # m/s² - minimum for reasonable signal
        
        # Check for non-numeric values
        if not all(isinstance(v, (int, float)) for v in gyro + accel):
            print("[!] Non-numeric IMU values detected; discarding message.")
            return False
            
        # Check for NaN or infinite values
        if any(np.isnan(v) or np.isinf(v) for v in gyro + accel):
            print("[!] NaN or infinite IMU values detected; discarding message.")
            return False
            
        # Check for zero values in all components (unlikely in real-world)
        if all(abs(v) < 1e-10 for v in gyro) and all(abs(v) < 1e-10 for v in accel):
            print("[!] All IMU values near zero; discarding message.")
            return False
            
        # Check for unreasonable values
        if any(abs(v) > MAX_GYRO for v in gyro):
            print(f"[!] Extreme gyro values detected: {gyro}; discarding.")
            return False
            
        if any(abs(v) > MAX_ACCEL for v in accel):
            print(f"[!] Extreme accel values detected: {accel}; discarding.")
            return False

        # Check for IMU initialization status
        accel_magnitude = np.sqrt(sum(a*a for a in accel))
        if not self.imu_initialized:
            if accel_magnitude < MIN_ACCEL:
                print(f"[!] Insufficient acceleration for IMU initialization: {accel_magnitude:.2f} m/s²")
                return False
            else:
                if len(self.imu_buffer) > 10:  # Need some history for initialization
                    self.imu_initialized = True
                    print("[+] IMU initialized successfully")
        
        # Check against moving average if available (detects sudden jumps)
        if self.imu_moving_avg:
            gyro_avg = np.mean([g["gyro"] for g in self.imu_moving_avg], axis=0)
            accel_avg = np.mean([a["accel"] for a in self.imu_moving_avg], axis=0)
            
            gyro_diff = np.sqrt(sum((g1-g2)**2 for g1, g2 in zip(gyro, gyro_avg)))
            accel_diff = np.sqrt(sum((a1-a2)**2 for a1, a2 in zip(accel, accel_avg)))
            
            # Thresholds for sudden changes (adjust based on your system dynamics)
            if gyro_diff > MAX_GYRO * 0.8:  # 80% of max as jump threshold
                print(f"[!] Sudden gyro jump detected: {gyro_diff:.2f} rad/s; discarding.")
                return False
                
            if accel_diff > MAX_ACCEL * 0.5:  # 50% of max as jump threshold
                print(f"[!] Sudden accel jump detected: {accel_diff:.2f} m/s²; discarding.")
                return False
                
        return True

    def create_imu_point(self, gyro, accel, ts):
        """Safely create an IMU point ensuring no NaN or zero values"""
        EPSILON = 1e-10
        
        # Apply epsilon with proper sign preservation
        def safe_value(v):
            if abs(v) < EPSILON:
                return EPSILON if v >= 0 else -EPSILON
            return v
        
        safe_gyro = [safe_value(g) for g in gyro]
        safe_accel = [safe_value(a) for a in accel]
        
        # Extra validation
        for i, g in enumerate(safe_gyro):
            if np.isnan(g) or np.isinf(g):
                print(f"Warning: After processing, gyro[{i}] is still invalid: {g}")
                safe_gyro[i] = EPSILON
        
        for i, a in enumerate(safe_accel):
            if np.isnan(a) or np.isinf(a):
                print(f"Warning: After processing, accel[{i}] is still invalid: {a}")
                safe_accel[i] = EPSILON
                
        # Create IMU point with validated values
        return orbslam3.IMU.Point(
            safe_gyro[0], safe_gyro[1], safe_gyro[2],
            safe_accel[0], safe_accel[1], safe_accel[2],
            ts
        )

    def debug_imu_measurements(self, imu_measurements):
        """Log detailed debug information about IMU measurements"""
        if len(imu_measurements) == 0:
            return
            
        # Only log every 100 frames to avoid console spam
        if int(time.time() * 10) % 100 == 0:
            print(f"Processing {len(imu_measurements)} IMU measurements")
            for i, imu in enumerate(imu_measurements[:3]):  # Print first few for debugging
                try:
                    # Use the correctly exposed attributes
                    print(f"  IMU[{i}]: t={imu.t:.6f}, " +
                        f"gyro=({imu.wx:.6f}, {imu.wy:.6f}, {imu.wz:.6f}), " +
                        f"accel=({imu.ax:.6f}, {imu.ay:.6f}, {imu.az:.6f})")
                
                    # Check for NaN values
                    try:
                        has_nan = (
                            np.isnan(imu.wx) or np.isnan(imu.wy) or np.isnan(imu.wz) or
                            np.isnan(imu.ax) or np.isnan(imu.ay) or np.isnan(imu.az) or
                            np.isnan(imu.t)
                        )
                        if has_nan:
                            print(f"  [!] WARNING: NaN values detected in IMU[{i}]")
                    except Exception as e:
                        print(f"  Error checking for NaN values in IMU[{i}]: {e}")
                    
                except Exception as e:
                    print(f"  Error accessing IMU[{i}] attributes: {e}")
    
    def _detect_map_reset(self):
        """Check if ORB-SLAM3 has reset its map using the C++ binding"""
        # Call the C++ method that detects resets
        reset_occurred = self.slam.was_map_reset()
        
        if reset_occurred:
            print("C++ detected an ORB-SLAM3 internal map reset!")
            return True
        
        # Additional check: Track state transitions (OK->RECENTLY_LOST->OK often indicates reset)
        current_state = self.slam.get_tracking_state()
        if hasattr(self, '_last_tracking_state'):
            if self._last_tracking_state == orbslam3.TrackingState.OK and current_state != orbslam3.TrackingState.OK:
                print(f"Tracking state changed from OK to {current_state}")
                if current_state == orbslam3.TrackingState.NOT_INITIALIZED:
                    print("Map reset detected from tracking state change to NOT_INITIALIZED")
                    self._last_tracking_state = current_state
                    return True
        
        # Update last tracking state
        self._last_tracking_state = current_state
        
        return False

    def _process_frame(self, frame, body, frame_ts_s, properties):
        """Process a frame with available IMU data."""
        try:
            start_time = time.time()

            # Create IMU measurements vector for this frame
            imu_measurements = []
            remaining_imu = []
            
            with self.processing_lock:
                # Sort IMU buffer by timestamp
                self.imu_buffer.sort(key=lambda x: x["timestamp_s"])
                
                valid_imu_count = 0
                for imu in self.imu_buffer:
                    if imu["timestamp_s"] <= frame_ts_s:
                        try:
                            # Extra validation before converting to orbslam3.IMU.Point
                            gyro = imu["gyro"]
                            accel = imu["accel"]
                            ts = imu["timestamp_s"]
                            
                            # Validate IMU values before processing
                            if not self.validate_imu_values(gyro, accel):
                                continue
                                
                            # Safely create IMU point
                            imu_point = self.create_imu_point(gyro, accel, ts)
                            imu_measurements.append(imu_point)
                            
                            # Publish processed IMU data
                            try:
                                processed_imu_data = {
                                    "timestamp": int(ts * 1e9),
                                    "gyroscope": {"x": gyro[0], "y": gyro[1], "z": gyro[2]},
                                    "accelerometer": {"x": accel[0], "y": accel[1], "z": accel[2]},
                                    "is_processed": True,
                                    "processed_by": "orbslam3",
                                    "used_with_frame_ts": frame_ts_s
                                }
                                
                                self.channel.basic_publish(
                                    exchange=PROCESSED_IMU_EXCHANGE,
                                    routing_key='',
                                    body=json.dumps(processed_imu_data)
                                )
                            except pika.exceptions.StreamLostError as e:
                                print(f"Stream lost during publish, attempting to reconnect: {e}")
                                self._attempt_recover_connection()
                            except Exception as e:
                                print(f"Error publishing processed IMU: {e}")
                            
                            valid_imu_count += 1
                        except Exception as e:
                            print(f"Error converting IMU data: {e}, skipping measurement")
                    else:
                        remaining_imu.append(imu)
                
                # Update IMU buffer with remaining measurements
                self.imu_buffer = remaining_imu
            
            # Proceed with SLAM processing
            if valid_imu_count >= self.MIN_IMU_COUNT:
                # Debug IMU measurements before processing
                self.debug_imu_measurements(imu_measurements)
                
                success = False
                processed = False
                
                try:
                    # Check for map reset before tracking
                    map_reset_before = self._detect_map_reset()
                    
                    # Process frame with IMU data
                    if self.slam_mode == "mono_inertial":
                        print(f"About to process frame at timestamp {frame_ts_s}")
                        print(f"IMU measurements count: {len(imu_measurements)}")
                        
                        for i, imu in enumerate(imu_measurements):
                            print(f"  IMU[{i}] before processing: t={imu.t:.6f}, "
                                f"wx={imu.wx:.6f}, wy={imu.wy:.6f}, wz={imu.wz:.6f}, "
                                f"ax={imu.ax:.6f}, ay={imu.ay:.6f}, az={imu.az:.6f}")
                        success = self.slam.process_image_mono_inertial(frame, frame_ts_s, imu_measurements)
                        processed = True
                    else:
                        success = self.slam.process_image_mono(frame, frame_ts_s)
                        processed = True
                        
                    # Check for map reset after tracking
                    map_reset_after = self._detect_map_reset()
                    
                    # If a reset occurred during this processing, handle it
                    if map_reset_before or map_reset_after:
                        print("Map reset detected during frame processing")
                        self._handle_map_reset()
                        
                except Exception as e:
                    print(f"SLAM processing error: {e}")
                
                # Update metrics and publish trajectory if processed
                if processed:
                    self._handle_processed_results(frame, frame_ts_s, success, body, start_time)
            else:
                print(f"Insufficient IMU measurements ({valid_imu_count} < {self.MIN_IMU_COUNT}), dropping frame")
        
        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()

    def _handle_map_reset(self):
        """Handle an ORB-SLAM3 map reset"""
        # Increment reset counters
        slam_reset_counter.inc()
        self.reset_count += 1
        
        # Store the last valid transformation for continuity
        if self.last_valid_transformation is not None:
            # If this is the first reset, just store the transformation
            if self.reset_transform is None:
                self.reset_transform = self.last_valid_transformation
            else:
                # For subsequent resets, chain the transformations
                self.reset_transform = np.matmul(self.reset_transform, self.last_valid_transformation)
            
            print(f"Stored transformation matrix for continuity across reset #{self.reset_count}")
            
        # Reset tracking counters but keep transformation
        self.failed_tracking_counter = 0
        self.restart_sent = False
        self.imu_initialized = False
        
        # Update tracking history with the reset event
        self.tracking_history.append(0)  # Count as tracking failure for metrics

    def _apply_transformation(self, pose):
        """Apply saved transformation to maintain trajectory continuity after resets"""
        if not isinstance(pose, np.ndarray):
            pose = np.array(pose)
            
        # Reshape to 4x4 if needed
        if pose.shape != (4, 4):
            pose = pose.reshape(4, 4)
            
        # If we have a reset transform, apply it
        if self.reset_transform is not None:
            transformed_pose = np.matmul(self.reset_transform, pose)
            return transformed_pose
        else:
            return pose

    def _handle_processed_results(self, frame, frame_ts_s, success, body, start_time):
        """Handle metrics and publishing for a processed frame with trajectory continuity."""
        try:
            # Update metrics
            slam_frames_processed.inc()
            elapsed = time.time() - start_time
            slam_frame_latency.observe(elapsed)
            slam_fps_gauge.set(1.0 / elapsed if elapsed > 0 else 0.0)
            
            # Update tracking success history
            self.tracking_history.append(1 if success else 0)
            if self.tracking_history:
                success_rate = 100.0 * sum(self.tracking_history) / len(self.tracking_history)
                slam_tracking_success_rate.set(success_rate)
            
            # Get the latest pose
            last_pose = self.slam.get_trajectory()
            if last_pose and len(last_pose) > 0:
                pose = last_pose[-1]
                pose_array = np.array(pose)
                
                # Store this pose as valid if tracking successful
                if success:
                    self.last_valid_transformation = pose_array.copy()
                
                # Apply transformations to maintain continuity if reset occurred
                transformed_pose = self._apply_transformation(pose_array)
                pose_list = transformed_pose.tolist()
                
                msg = {
                    "timestamp_ns": int(frame_ts_s * 1e9),
                    "pose": pose_list,
                    "tracking_success": success,
                    "reset_count": self.reset_count
                }
                
                try:
                    self.channel.basic_publish(
                        exchange=TRAJECTORY_DATA_EXCHANGE,
                        routing_key='',
                        body=json.dumps(msg)
                    )
                    
                    combined_msg = {
                        "timestamp_ns": int(frame_ts_s * 1e9),
                        "pose": pose_list,
                        "tracking_success": success,
                        "reset_count": self.reset_count,
                        "image_b64": base64.b64encode(body).decode('utf-8')
                    }
                    
                    self.channel.basic_publish(
                        exchange=SLAM_DATA_EXCHANGE,
                        routing_key='',
                        body=json.dumps(combined_msg)
                    )
                except pika.exceptions.StreamLostError as e:
                    print(f"Stream lost during publish, attempting to reconnect: {e}")
                    self._attempt_recover_connection()
                except Exception as e:
                    print(f"Error publishing trajectory data: {e}")
            
            # Track tracking failures
            if not success:
                self.failed_tracking_counter += 1
                if self.failed_tracking_counter >= self.RESET_THRESHOLD and not self.restart_sent:
                    print(f"Tracking failure threshold reached ({self.failed_tracking_counter}), sending restart command")
                    try:
                        restart_msg = json.dumps({"type": "restart"})
                        self.channel.basic_publish(
                            exchange=RESTART_EXCHANGE,
                            routing_key='',
                            body=restart_msg
                        )
                        self.restart_sent = True
                    except Exception as e:
                        print(f"Error sending restart command: {e}")
            else:
                self.failed_tracking_counter = 0
                self.restart_sent = False
        except Exception as e:
            print(f"Error in _handle_processed_results: {e}")
            import traceback
            traceback.print_exc()

    def _attempt_recover_connection(self):
        """Attempt to recover RabbitMQ connection after failure"""
        try:
            # Only attempt recovery if connection appears closed
            if not self.connection.is_open:
                print("Attempting to reconnect to RabbitMQ...")
                
                try:
                    # Close the old connection if possible
                    if self.connection:
                        try:
                            self.connection.close()
                        except:
                            pass
                except:
                    pass
                
                # Create a new connection and channel
                self._setup_rabbitmq()
                
                print("RabbitMQ connection recovered")
        except Exception as e:
            print(f"Failed to recover RabbitMQ connection: {e}")

    def on_image_message(self, ch, method, properties, body):
        try:
            # Decode the image
            np_arr = np.frombuffer(body, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Frame timestamp in seconds
            frame_ts_ns = int(properties.headers.get("timestamp_ns"))
            frame_ts_s = frame_ts_ns / 1e9
            
            with self.processing_lock:
                # Add to timestamp history
                self.last_image_timestamps.append(frame_ts_s)
                
                # Keep only the last N timestamps
                if len(self.last_image_timestamps) > 10:
                    self.last_image_timestamps.pop(0)
                
                # Count IMU measurements available for this frame
                imu_count = self._count_imu_for_frame(frame_ts_s)
                
                if imu_count >= self.MIN_IMU_COUNT:
                    # Sufficient IMU measurements available, process immediately
                    # Release lock before processing to prevent deadlock
                    frame_copy = frame.copy()
                    self.processing_lock.release()
                    try:
                        self._process_frame(frame_copy, body, frame_ts_s, properties)
                    finally:
                        # Re-acquire lock
                        self.processing_lock.acquire()
                else:
                    # Not enough IMU data, add to buffer if space available
                    if len(self.frame_buffer) < self.MAX_FRAME_BUFFER_SIZE:
                        self.frame_buffer.append({
                            "frame": frame,
                            "body": body,
                            "timestamp": frame_ts_s,
                            "properties": properties,
                            "buffer_time": time.time()
                        })
                    else:
                        # Buffer full, drop oldest frame
                        self.frame_buffer.pop(0)
                        print(f"Frame buffer full, dropping oldest frame to make room")
                        self.frame_buffer.append({
                            "frame": frame,
                            "body": body,
                            "timestamp": frame_ts_s,
                            "properties": properties,
                            "buffer_time": time.time()
                        })
        
        except Exception as e:
            print(f"Error in on_image_message: {e}")
            import traceback
            traceback.print_exc()

    def on_imu_message(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            # Check that the required keys exist
            if 'timestamp' not in data or 'accelerometer' not in data or 'gyroscope' not in data:
                print("[!] IMU message missing required keys; discarding message.")
                return

            # Extract IMU values
            accel_x = data['accelerometer'].get('x', 0)
            accel_y = data['accelerometer'].get('y', 0)
            accel_z = data['accelerometer'].get('z', 0)
            gyro_x = data['gyroscope'].get('x', 0)
            gyro_y = data['gyroscope'].get('y', 0)
            gyro_z = data['gyroscope'].get('z', 0)
            
            # Collect values for validation
            gyro = [gyro_x, gyro_y, gyro_z]
            accel = [accel_x, accel_y, accel_z]
            
            # Enhanced validation for IMU values
            if not self.validate_imu_values(gyro, accel):
                return
                    
            imu_ts = data['timestamp']  # timestamp in nanoseconds
            
            with self.processing_lock:
                imu_point = {
                    "timestamp_s": imu_ts / 1e9,  # Convert nanoseconds to seconds
                    "gyro": gyro,  # Gyroscope data in rad/s
                    "accel": accel  # Accelerometer data in m/s²
                }
                
                # More lenient timestamp filtering - keep IMU data within reasonable time range
                if len(self.last_image_timestamps) >= 4:
                    max_image_ts = max(self.last_image_timestamps)
                    min_image_ts = min(self.last_image_timestamps)
                    
                    # Allow IMU data slightly in the future and up to 2 seconds in the past
                    if (imu_point["timestamp_s"] > max_image_ts + 0.5) or (imu_point["timestamp_s"] < min_image_ts - 2.0):
                        return  # Discard this IMU measurement
                
                # Add to buffer if it's newer than the last one or if buffer is empty
                if not self.imu_buffer or imu_point["timestamp_s"] > self.imu_buffer[-1]["timestamp_s"]:
                    self.imu_buffer.append(imu_point)
                    
                    # Update moving average window for IMU smoothing (for jump detection)
                    self.imu_moving_avg.append(imu_point)
                    if len(self.imu_moving_avg) > 10:  # Keep window size manageable
                        self.imu_moving_avg.pop(0)
                
                # Prevent buffer from growing too large by trimming old measurements
                MAX_IMU_BUFFER = 1000  # Maximum number of IMU measurements to keep
                if len(self.imu_buffer) > MAX_IMU_BUFFER:
                    # Remove oldest IMU measurements, keeping the last MAX_IMU_BUFFER
                    self.imu_buffer = self.imu_buffer[-MAX_IMU_BUFFER:]
                    
                # Check if any buffered frames can now be processed with this new IMU data
                self._check_buffered_frames_after_imu()

        except Exception as e:
            print(f"Error in on_imu_message: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_buffered_frames_after_imu(self):
        """Check if any buffered frames have enough IMU data after new IMU arrives"""
        try:
            # We're already holding the processing_lock in the calling context
            i = 0
            while i < len(self.frame_buffer):
                frame_info = self.frame_buffer[i]
                frame_ts_s = frame_info["timestamp"]
                
                imu_count = self._count_imu_for_frame(frame_ts_s)
                
                if imu_count >= self.MIN_IMU_COUNT:
                    # Sufficient IMU data available, remove from buffer
                    frame_info = self.frame_buffer.pop(i)
                    frame = frame_info["frame"].copy()  # Make a copy for thread safety
                    
                    # Release lock before processing
                    self.processing_lock.release()
                    try:
                        self._process_frame(
                            frame, 
                            frame_info["body"], 
                            frame_info["timestamp"],
                            frame_info["properties"]
                        )
                    finally:
                        # Re-acquire lock
                        self.processing_lock.acquire()
                    
                    # Start over since we modified the buffer
                    i = 0
                else:
                    i += 1
        except Exception as e:
            print(f"Error in _check_buffered_frames_after_imu: {e}")
            import traceback
            traceback.print_exc()

    def _handle_restart(self, ch, method, properties, body):
        """Handle restart commands with trajectory continuity"""
        try:
            msg = json.loads(body)
            if msg.get("type") == "restart":
                print("Restart command received in SLAM. Preserving trajectory...")
                
                # Store the last valid transformation for continuity
                if self.last_valid_transformation is not None:
                    # If this is the first reset, just store the transformation
                    if self.reset_transform is None:
                        self.reset_transform = self.last_valid_transformation
                    else:
                        # For subsequent resets, chain the transformations
                        self.reset_transform = np.matmul(self.reset_transform, self.last_valid_transformation)
                    
                    print(f"Stored transformation matrix for continuity across reset #{self.reset_count+1}")
                
                # Increment counters
                slam_reset_counter.inc()
                self.reset_count += 1
                
                # Perform the reset
                self.slam.reset()
                
                # Reset tracking counters but keep transformation
                self.failed_tracking_counter = 0
                self.restart_sent = False
                self.imu_initialized = False
                
                # Clear buffers
                with self.processing_lock:
                    self.imu_buffer.clear()
                    self.frame_buffer.clear()
                    self.imu_moving_avg.clear()
                    
                print("SLAM system reset complete, trajectory continuity preserved")
                
        except Exception as e:
            print("Error handling restart in SLAM:", e)
            import traceback
            traceback.print_exc()

    def run_forever(self):
        print(" [*] Starting blocking consume. Press Ctrl+C to stop.")
        while True:
            try:
                self.channel.start_consuming()
            except KeyboardInterrupt:
                print("\n [x] Interrupted by user")
                break
            except pika.exceptions.ConnectionClosedByBroker:
                print("Connection closed by broker, retrying...")
                time.sleep(5)
                self._attempt_recover_connection()
            except pika.exceptions.AMQPChannelError as e:
                print(f"Channel error: {e}, retrying...")
                time.sleep(5)
                self._attempt_recover_connection()
            except pika.exceptions.AMQPConnectionError:
                print("Connection was closed, retrying...")
                time.sleep(5)
                self._attempt_recover_connection()
            except Exception as e:
                print(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                self._attempt_recover_connection()
        
        # Cleanup
        if self.slam:
            self.slam.shutdown()
        if self.connection and self.connection.is_open:
            self.connection.close()

if __name__ == "__main__":
    start_http_server(8000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=os.getenv("SLAM_MODE", "mono_inertial"),
                        choices=["mono", "mono_inertial"],
                        help="SLAM mode to run: 'mono' or 'mono_inertial'")
    args = parser.parse_args()
    node = RunRGBD(slam_mode=args.mode)
    node.run_forever()