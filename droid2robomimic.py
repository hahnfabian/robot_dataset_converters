""""
Port DROID (https://droid-dataset.github.io/) dataset to robomimic.

1. DROID style:
    /episode/
        episode_metadata/
            ...
        steps/
            is_first
            is_last
            is_terminal
            language_instruction
            language_instruction_2
            language_instruction_3
            observation/  --> current state
                gripper_position (1,)
                cartesian_position (6,)
                joint_position (7,)
                wrist_image_left (180, 320, 3)
                exterior_image_1_left (180, 320, 3)
                exterior_image_2_left (180, 320, 3)
            action_dict/  --> commanded state
                gripper_position (1,)
                gripper_velocity (1,)
                cartesian_position (6,)
                cartesian_velocity (6,)
                joint_position (7,)
                joint_velocity (7,)
            reward
            action (7,) --> [6x joint velocities, 1x gripper position]


2. Robomimic style:
    /data/
        demo_0/
            num_samples --> number of state-action samples 
            actions         (T, action_dim) --> [x, y, z, roll, pitch, yaw, gripper] 
            obs/
                agentview_image (T, H, W, C)
                robot0_eef_pos  (T, 3) --> [x,y,z]
                robot0_eef_quat
                robot0_eef_vel_ang
                robot0_eef_vel_lin
                robot0_eye_in_hand_image
                robot0_gripper_qpos
                robot0_gripper_qvel
                robot0_joint_pos
                robot0_joint_pos_cos
                robot0_joint_pos_sin
                robot0_joint_vel
            rewards         (T,)
            dones           (T,)  --> is 1 corresponding action in the state should terminate the episode
        demo_1/
            ...
    /mask/
        train             episode indices for train split
        valid             episode indices for valid split
"""


import h5py
import argparse
import tensorflow_datasets as tfds
import numpy as np
import os
import sys
import json
from tqdm import tqdm
import cv2


CAMERA_HEIGHT = 230
CAMERA_WIDTH = 180

# TODO replace with the normalised a_min, a_max (gotten by the DataLoader?)
CONTROLLER_OUTPUT_MAX = [1.0]*7
CONTROLLER_OUTPUT_MIN = [-1.0]*7


env_args = {
    "env_name": "DROID_100",
    "type": 1,
    "robots": [
        "Panda"
    ],
    "camera_depths": False,
    "camera_heights": CAMERA_HEIGHT,    
    "camera_widths": CAMERA_WIDTH,
    "reward_shaping": False,
    "camera_names": [
            "agentview",
    ],
    "env_kwargs": {
        "has_renderer": False, 
        "has_offscreen_renderer": True, 
        "ignore_done": True, # Check
        "use_object_obs": False, # Check
        "use_camera_obs": True, 
        "control_freq": 20, # Check
        "controller_configs": {
            "type": "JOINT_VELOCITIES",
            "input_max": 1,
            "input_min": -1,
            "output_max": CONTROLLER_OUTPUT_MAX,
            "output_min": [-1.0]*7,
            "kp": 150, # CHECK all the following -->
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [
                0,
                300
            ],
            "damping_limits": [
                0,
                10
            ],
            # "position_limits": null,
            # "orientation_limits": null,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.0 # Check
        },
        "render_gpu_device_id": 0,
    }
}

def resize_with_padding(img, size=256):
    h, w = img.shape[:2]
    
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    return canvas


def convert_episode(episode):

    T = len(episode["steps"])

    joint_positions = np.empty((T, 7), dtype=np.float32)
    cartesian_positions = np.empty((T, 6), dtype=np.float32)
    gripper_positions = np.empty((T, 1), dtype=np.float32)
    agentview_images = np.empty((T, CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    actions = np.empty((T, 7), dtype=np.float32)
    dones = np.empty((T,), dtype=np.bool_)

    steps = list(episode["steps"])
    first_step = steps[0]

    language_instruction = first_step["language_instruction"].numpy().decode("utf-8")
    language_instruction_2 = first_step["language_instruction_2"].numpy().decode("utf-8")
    language_instruction_3 = first_step["language_instruction_3"].numpy().decode("utf-8")


    for i, step in enumerate(episode["steps"]):      
        obs = step["observation"]

        joint_positions[i] = obs["joint_position"].numpy().astype(np.float32)  
        cartesian_positions[i] = obs["cartesian_position"].numpy().astype(np.float32)  
        gripper_positions[i] = obs["gripper_position"].numpy().astype(np.float32) 
        
        # TODO resize the images?
        img = obs["exterior_image_1_left"].numpy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # img = resize_with_padding(img)
        agentview_images[i] = img

        dones[i] = step["is_terminal"].numpy().astype(np.bool_)
        actions[i] = step["action"].numpy()    

    ep = {
        "obs": {
            "joint_position": joint_positions,
            "cartesian_position": cartesian_positions,
            "gripper_position": gripper_positions,
            "agentview_image": agentview_images,
        },
        "language_instruction": language_instruction,
        "language_instruction_2": language_instruction_2,
        "language_instruction_3": language_instruction_3,
        "actions": actions,
        "dones": dones,
    }

    return ep


def write_episode(h5, idx, ep):

    next_obs = {}
    for k, v in ep["obs"].items():
        next_v = np.concatenate([v[1:], v[-1:]], axis=0)
        next_obs[k] = next_v

    ep["next_obs"] = next_obs

    g = h5["data"].create_group(f"demo_{idx}")
    
    # Create obs group
    obs_group = g.create_group("obs")
    next_obs_group = g.create_group("next_obs")
    
    # Write all observations
    for k, v in ep["obs"].items():
        obs_group.create_dataset(k, data=v, compression="gzip")
    
    for k, v in ep["next_obs"].items():
        next_obs_group.create_dataset(k, data=v, compression="gzip")

    g.create_dataset("actions", data=ep["actions"], compression="gzip")
    g.create_dataset("dones", data=ep["dones"], compression="gzip")

    g.attrs["num_samples"] = ep["actions"].shape[0]
    g.attrs["language_instruction"] = ep["language_instruction"]
    g.attrs["language_instruction_2"] = ep["language_instruction_2"]
    g.attrs["language_instruction_3"] = ep["language_instruction_3"]

# --- not needed if the DataLoader handles the normalisation correctly ---

def compute_action_stats(ds, num_episodes=None):
    mins = []
    maxs = []

    for i, episode in enumerate(ds):
        if num_episodes and i >= num_episodes:
            break

        acts = np.array([step["action"].numpy() for step in episode["steps"]])
        mins.append(acts.min(axis=0))
        maxs.append(acts.max(axis=0))

    global_min = np.min(np.stack(mins), axis=0)
    global_max = np.max(np.stack(maxs), axis=0)

    return global_min, global_max

def normalise(actions, a_min, a_max):
    denom = (a_max - a_min)
    denom[denom == 0] = 1.0
    return 2 * (actions - a_min) / denom - 1

# --- end of normalisation code ---

def main(data_dir, output_path, num_episodes):
    print(f"...Loading dataset from {data_dir}")
    
    builder = tfds.builder("droid_100", data_dir=data_dir)
    ds = builder.as_dataset(split="train", shuffle_files=False)
    
    print("...Dataset loaded, starting conversion")
    
    with h5py.File(output_path, "w") as h5:
        h5.create_group("data")
        
        h5["data"].attrs["env_args"] = json.dumps(env_args)

        ep_idx = 0
        episode_indices = []
        
        # TODO: only write after N eps
        for episode in tqdm(ds):
            if num_episodes is not None and ep_idx >= num_episodes:
                break
                
            try:
                robomimic_ep = convert_episode(episode)
                
                write_episode(h5, ep_idx, robomimic_ep)
                
                episode_indices.append(ep_idx)
                
                # print(f"...Wrote episode {ep_idx} with {len(robomimic_ep['actions'])} steps")
                ep_idx += 1
                
            except Exception as e:
                print(f"[ERROR] Error processing episode {ep_idx}: {e}")
                continue
        
        # Create train/valid split mask (80/20 split)
        num_episodes_total = ep_idx
        num_train = int(0.8 * num_episodes_total)
        
        # Shuffle indices for random split
        np.random.seed(42)  # For reproducibility
        shuffled_indices = episode_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        train_indices = sorted(shuffled_indices[:num_train])
        val_indices = sorted(shuffled_indices[num_train:])
        train_demo_names = [f"demo_{i}" for i in train_indices]
        val_demo_names = [f"demo_{i}" for i in val_indices]
        # Create mask group
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

        #TODO: add env_metadata
        
        print(f"\nConversion complete!")
        print(f"...Total episodes: {num_episodes_total}")
        print(f"...Train episodes: {len(train_indices)}")
        print(f"...Validation episodes: {len(val_indices)}")
        print(f"...Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DROID dataset to robomimic format")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Input data directory",
        default="gs://gresearch/robotics",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="robomimic_dataset.hdf5",
        help="Output HDF5 file path"
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to convert (default: all)"
    )
    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output) or "."
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)
    
    main(args.data_dir, args.output, args.num_episodes)