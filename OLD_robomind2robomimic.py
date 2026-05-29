"""
Convert RoboMind episodes to a robomimic / Seeker-compatible HDF5 dataset.

├── language_distilbert  shape=(1, 1, 768) dtype=float16
├── language_raw  shape=(1,) dtype=object
├── master/
│   ├── joint_position  shape=(168, 8) dtype=float64
├── observations/
│   ├── depth_images/
│   │   ├── camera_left  shape=(168,) dtype=object
│   │   ├── camera_right  shape=(168,) dtype=object
│   │   ├── camera_top  shape=(168,) dtype=object
│   ├── rgb_images/
│   │   ├── camera_left  shape=(168,) dtype=object
│   │   ├── camera_right  shape=(168,) dtype=object
│   │   ├── camera_top  shape=(168,) dtype=object
├── puppet/
│   ├── end_effector  shape=(168, 6) dtype=float64
│   ├── joint_position  shape=(168, 8) dtype=float64

"""



import h5py
import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
import cv2
from pathlib import Path
import json


ENV_ARGS = {
    "env_name": "pickup_pear_from_bowl",
    "type": 1,  # 1 = robosuite-style env

    "env_kwargs": {
        "robots": ["Panda"],

        "has_renderer": False,
        "has_offscreen_renderer": True,   
        "ignore_done": True,

        "use_object_obs": False,
        "use_camera_obs": True,

        "camera_names": [
            "agentview",
            "agentview_left",
            "agentview_right",
        ],

        "camera_heights": [256, 256, 256],
        "camera_widths": [256, 256, 256],
        "camera_depths": [False, False, False],

        "control_freq": 20,

        "controller_configs": {
            "type": "JOINT_POSITION",  

            "input_max": 1.0,
            "input_min": -1.0,

            "output_max": [1.0] * 8,
            "output_min": [-1.0] * 8,

            "kp": 150,
            "damping": 1,

            "impedance_mode": "fixed",

            "kp_limits": [0, 300],
            "damping_limits": [0, 10],

            "uncouple_pos_ori": True,
            "control_delta": False, 

            "interpolation": None,
            "ramp_ratio": 0.0,
        },

        "render_gpu_device_id": 0,
    },
}


def decode_image(img_bytes):
    img_array = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def rotvec_to_matrix(rotvec):
    mat, _ = cv2.Rodrigues(rotvec.astype(np.float64))
    return mat.astype(np.float32)

def euler_xyz_to_matrix(euler_xyz):
    rx, ry, rz = euler_xyz.astype(np.float64)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def matrix_to_quat_xyzw(mat):
    m = mat.astype(np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    n = np.linalg.norm(quat)
    return quat / max(n, 1e-8)


def quat_xyzw_to_matrix(quat_xyzw):
    x, y, z, w = quat_xyzw.astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def orientation_to_matrix(orientation, orientation_format):
    if orientation_format == "rotvec":
        return rotvec_to_matrix(orientation)
    if orientation_format == "euler_xyz":
        return euler_xyz_to_matrix(orientation)
    if orientation_format == "quat_xyzw":
        return quat_xyzw_to_matrix(orientation)
    if orientation_format == "quat_wxyz":
        return quat_xyzw_to_matrix(np.array([orientation[1], orientation[2], orientation[3], orientation[0]], dtype=np.float32))
    if orientation_format == "identity":
        return np.eye(3, dtype=np.float32)
    raise ValueError(f"Unsupported orientation_format: {orientation_format}")


def extract_gripper_qpos(joint_pos):
    if joint_pos.shape[1] >= 2:
        if joint_pos.shape[1] == 8:
            g = joint_pos[:, -1:]
            return np.concatenate([g, g], axis=1).astype(np.float32)
        return joint_pos[:, -2:].astype(np.float32)
    return np.zeros((joint_pos.shape[0], 2), dtype=np.float32)


def read_episode(input_path, orientation_format="euler_xyz"):
    with h5py.File(input_path, "r") as f:

        T = f["master/joint_position"].shape[0]

        language_raw = f["language_raw"][0].decode("utf-8")

        actions = f["master/joint_position"][:].astype(np.float32)
        
        robot0_joint_pos = f["puppet/joint_position"][:].astype(np.float32)
        robot0_eef = f["puppet/end_effector"][:].astype(np.float32)
        robot0_gripper_qpos = extract_gripper_qpos(robot0_joint_pos)

        robot0_eef_pos = robot0_eef[:, :3]
        robot0_eef_ori = robot0_eef[:, 3:]
        robot0_eef_rot = np.zeros((T, 9), dtype=np.float32)
        robot0_eef_quat = np.zeros((T, 4), dtype=np.float32)

        for i in range(T):
            rot = orientation_to_matrix(robot0_eef_ori[i], orientation_format=orientation_format)
            robot0_eef_rot[i] = rot.reshape(-1)
            robot0_eef_quat[i] = matrix_to_quat_xyzw(rot)

        rgb_top = f["observations/rgb_images/camera_top"]
        rgb_left = f["observations/rgb_images/camera_left"]
        rgb_right = f["observations/rgb_images/camera_right"]


        agentview_image = []
        agentview_left_image = []
        agentview_right_image = []

        for i in range(T):
            img = decode_image(rgb_top[i])
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            agentview_image.append(img)

            img_left = decode_image(rgb_left[i])
            img_left = cv2.resize(img_left, (256, 256))
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            agentview_left_image.append(img_left)

            img_right = decode_image(rgb_right[i])
            img_right = cv2.resize(img_right, (256, 256))
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
            agentview_right_image.append(img_right)

        agentview_image = np.stack(agentview_image, axis=0).astype(np.uint8)
        agentview_left_image = np.stack(agentview_left_image, axis=0).astype(np.uint8)
        agentview_right_image = np.stack(agentview_right_image, axis=0).astype(np.uint8)

        dones = np.zeros((T,), dtype=np.bool_)
        dones[-1] = True

        rewards = np.zeros((T,), dtype=np.float32)

    ep = {
        "obs": {
            "robot0_joint_pos": robot0_joint_pos,
            "robot0_gripper_qpos": robot0_gripper_qpos,
            "robot0_eef_pos": robot0_eef_pos,
            "robot0_eef_rot": robot0_eef_rot,
            "robot0_eef_quat": robot0_eef_quat,
            "agentview_image": agentview_image,
            "agentview_left_image": agentview_left_image,
            "agentview_right_image": agentview_right_image,
        },
        "language_raw": language_raw,
        "actions": actions,
        "absolute_actions": actions.copy(),
        "rewards": rewards,
        "dones": dones,
    }

    return ep


def add_next_obs(ep):
    next_obs = {}
    for k, v in ep["obs"].items():
        next_v = np.concatenate([v[1:], v[-1:]], axis=0)
        next_obs[k] = next_v
    ep["next_obs"] = next_obs
    return ep


def write_episode(h5, idx, ep):
    g = h5["data"].create_group(f"demo_{idx}")

    obs_group = g.create_group("obs")
    next_obs_group = g.create_group("next_obs")

    for k, v in ep["obs"].items():
        obs_group.create_dataset(k, data=v, compression="gzip")

    for k, v in ep["next_obs"].items():
        next_obs_group.create_dataset(k, data=v, compression="gzip")

    g.create_dataset("actions", data=ep["actions"], compression="gzip")
    g.create_dataset("absolute_actions", data=ep["absolute_actions"], compression="gzip")
    g.create_dataset("rewards", data=ep["rewards"], compression="gzip")
    g.create_dataset("dones", data=ep["dones"], compression="gzip")

    g.attrs["num_samples"] = ep["actions"].shape[0]
    g.attrs["task_instruction"] = ep["language_raw"]


def main(data_dir, output_path, num_episodes, orientation_format):
    base_dir = Path(data_dir)
    files = list(base_dir.glob("success_episodes/train/*/data/trajectory.hdf5"))

    print(f"... Found {len(files)} episodes")

    if num_episodes is not None:
        files = files[:num_episodes]
        print(f"... Slicing the first {num_episodes} episodes")

    min_a, max_a = compute_action_min_max(files)

    with h5py.File(output_path, "w") as h5:
        h5.create_group("data")
        h5["data"].attrs["env_args"] = json.dumps(ENV_ARGS)
        h5["data"].attrs["action_min"] = min_a.tolist()
        h5["data"].attrs["action_max"] = max_a.tolist()
        h5["data"].attrs["source"] = "RoboMind"
        h5["data"].attrs["orientation_format"] = orientation_format

        ep_idx = 0
        episode_indices = []


        for path in tqdm(files, desc="Converting episodes"):

            try:
                ep = read_episode(path, orientation_format=orientation_format)
                ep = add_next_obs(ep)
                write_episode(h5, ep_idx, ep)

                episode_indices.append(ep_idx)
                ep_idx += 1

            except Exception as e:
                print(f"[ERROR] {path}: {e}")
        
        num_episodes_total = ep_idx
        num_train = int(0.8 * num_episodes_total)

        np.random.seed(42)  # For reproducibility
        shuffled_indices = episode_indices.copy()
        np.random.shuffle(shuffled_indices)

        train_indices = sorted(shuffled_indices[:num_train])
        val_indices = sorted(shuffled_indices[num_train:])
        train_demo_names = [f"demo_{i}" for i in train_indices]
        val_demo_names = [f"demo_{i}" for i in val_indices]

        mask_group = h5.create_group("mask")
        mask_group.create_dataset(
            "train",
            data=np.array(train_demo_names, dtype="S"),  # S = fixed-length bytes
            compression="gzip"
        )
        mask_group.create_dataset(
            "valid",
            data=np.array(val_demo_names, dtype="S"),
            compression="gzip"
        )


    print(f"... Wrote {ep_idx} episodes to {output_path}")
    print(f"...Total episodes: {num_episodes_total}")
    print(f"...Train episodes: {len(train_indices)}")
    print(f"...Validation episodes: {len(val_indices)}")
        

def compute_action_min_max(files):
    min_a = None
    max_a = None

    for path in tqdm(files, desc="Computing action min/max"):
        try:
            with h5py.File(path, "r") as f:
                actions = f["master/joint_position"][:].astype(np.float32)

                if min_a is None:
                    min_a = actions.min(axis=0)
                    max_a = actions.max(axis=0)
                else:
                    min_a = np.minimum(min_a, actions.min(axis=0))
                    max_a = np.maximum(max_a, actions.max(axis=0))

        except Exception as e:
            print(f"[ERROR stats] {path}: {e}")

    return min_a, max_a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboMind dataset to robomimic format")
    
    parser.add_argument(
        "--data-dir",
        "-i",
        type=str,
        help="Input data directory (e.g., ~/research/241023_pick_pear_from_bowl_1)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="converted_robomind_to_robomimic_dataset.hdf5",
        help="Output HDF5 file path"
    )
    
    parser.add_argument(
        "--num-episodes",
        "-n",
        type=int,
        default=None,
        help="Number of episodes to convert (default: all)"
    )
    
    parser.add_argument(
        "--orientation-format",
        type=str,
        default="euler_xyz",
        choices=["rotvec", "euler_xyz", "quat_xyzw", "quat_wxyz", "identity"],
        help="How puppet/end_effector[:, 3:] is interpreted before writing eef_rot/eef_quat.",
    )

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output) or "."
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)
    
    main(args.data_dir, args.output, args.num_episodes, args.orientation_format)