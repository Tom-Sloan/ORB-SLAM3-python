"""
Shared Memory Format Documentation

Two shared memory blocks are used to share SLAM trajectory data:

1. 'slam_trajectory' - Stores pose and timestamp data
   - Shape: (MAX_POSES, 5, 4) as np.float64
   - Size: MAX_POSES * (16 * 8 + 8) bytes
   - Structure for each pose entry [i]:
     trajectory_array[i, :4, :] = 4x4 transformation matrix (SE3 pose)
     trajectory_array[i, 4, 0] = timestamp (float64)
     trajectory_array[i, 4, 1:] = unused

2. 'slam_trajectory_meta' - Stores metadata
   - Shape: (2,) as np.int64
   - Size: 16 bytes
   - Structure:
     meta_array[0] = total number of poses stored (up to MAX_POSES)
     meta_array[1] = current write position (for circular buffer)

Circular Buffer Operation:
- New poses are written at position meta_array[1]
- Write position wraps around to 0 after reaching MAX_POSES
- When buffer is full, oldest poses are overwritten
- Reader can track write_pos to detect new poses

Example Usage:
    # Writer (SLAM system)
    write_pos = meta_array[1]
    trajectory_array[write_pos, :4, :] = new_pose_matrix
    trajectory_array[write_pos, 4, 0] = timestamp
    meta_array[1] = (write_pos + 1) % MAX_POSES
    meta_array[0] = min(meta_array[0] + 1, MAX_POSES)

    # Reader
    write_pos = meta_array[1]
    if write_pos != last_read_pos:
        latest_pose = trajectory_array[write_pos-1].copy()
        latest_timestamp = latest_pose[4, 0]
""" 