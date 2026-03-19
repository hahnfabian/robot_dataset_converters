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
import tqdm

env_args = {
    "env_name": "DROID_100",
    "type": 1,
    "robots": [
        "Panda"
    ],
    "camera_depths": False,
    "camera_heights": 180,    
    "camera_widths": 320,
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
            "output_max": [1.0]*7,
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



def convert_episode(episode):
    cartesian_positions = []
    gripper_positions = []
    joint_positions = []
    
    agentview_images = []
    
    actions = []
    dones = []

    language_instruction = []



    for step in episode["steps"]:      
        language_instruction.append(step["language_instruction"])

        # Extract observations
        joint_position = step["observation"]["joint_position"].numpy().astype(np.float32)  
        cartesian_position = step["observation"]["cartesian_position"].numpy().astype(np.float32)  
        gripper_position = step["observation"]["gripper_position"].numpy().astype(np.float32) 
        
        joint_positions.append(joint_position)
        cartesian_positions.append(cartesian_position)  
        gripper_positions.append(gripper_position)        

        # TODO resize the images?
        img = step["observation"]["exterior_image_1_left"].numpy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        agentview_images.append(img)
               
        dones.append(step["is_terminal"].numpy())
        actions.append(step["action"].numpy())

    joint_velocity_max = np.array([3.1415]*6 + [1.0])
    joint_velocity_min = -joint_velocity_max
    actions_norm = 2 * (actions - joint_velocity_min) / (joint_velocity_max - joint_velocity_min) - 1

    # Stack all arrays
    ep = {
        "obs": {
            "joint_position": np.stack(joint_positions, axis=0).astype(np.float32),
            "cartesian_position": np.stack(cartesian_positions, axis=0).astype(np.float32),
            "gripper_position": np.stack(gripper_positions, axis=0).astype(np.float32),
            "agentview_image": np.stack(agentview_images, axis=0).astype(np.uint8),
        },
        "language_instructions": np.stack(language_instruction, axis=0),
        "actions": np.stack(actions_norm, axis=0).astype(np.float32),
        "dones": np.stack(dones, axis=0).astype(np.bool_),
    }
    
    return ep


def write_episode(h5, idx, ep):
    g = h5["data"].create_group(f"demo_{idx}")
    
    # Create obs group
    obs_group = g.create_group("obs")
    
    # Write all observations
    for k, v in ep["obs"].items():
        obs_group.create_dataset(k, data=v, compression="gzip")
    
    g.create_dataset("actions", data=ep["actions"], compression="gzip")
    g.create_dataset("language_instructions", data=ep["language_instructions"], compression="gzip") 
    g.create_dataset("dones", data=ep["dones"], compression="gzip")

    g.attrs["num_samples"] = ep["actions"].shape[0]


def main(data_dir, output_path, num_episodes):
    print(f"...Loading dataset from {data_dir}")
    
    builder = tfds.builder("droid_100", data_dir=data_dir)
    ds = builder.as_dataset(split="train", shuffle_files=False)
    
    print("...Dataset loaded, starting conversion")
    
    with h5py.File(output_path, "w") as h5:
        h5.create_group("data")
        
        h5["data"].attrs["env_args"] = np.string_(json.dumps(env_args))

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
                
                print(f"...Wrote episode {ep_idx} with {len(robomimic_ep['actions'])} steps")
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
        valid_indices = sorted(shuffled_indices[num_train:])
        train_demo_names = [f"demo_{i}" for i in train_indices]
        valid_demo_names = [f"demo_{i}" for i in valid_indices]
        # Create mask group
        mask_group = h5.create_group("mask")
        mask_group.create_dataset(
        "train",
        data=np.array(train_demo_names, dtype="S"),  # S = fixed-length bytes
        compression="gzip"
        )
        mask_group.create_dataset(
            "valid",
            data=np.array(valid_demo_names, dtype="S"),
            compression="gzip"
        )

        #TODO: add env_metadata
        
        print(f"\nConversion complete!")
        print(f"...Total episodes: {num_episodes_total}")
        print(f"...Train episodes: {len(train_indices)}")
        print(f"...Valid episodes: {len(valid_indices)}")
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