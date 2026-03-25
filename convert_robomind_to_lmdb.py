# TODO also process val data

"""
Convert raw RoboMind (https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) hdf5 episodes into Seeker's LMDB cache format.

Expected raw RoboMind episode layout:

<task_name>/success_episodes/{train,val}/<episode_id>/data/trajectory.hdf5
├── language_distilbert                             shape=(1, 1, 768) dtype=float16
├── language_raw                                    shape=(1,) dtype=object
├── master/joint_position                           (T, 8)
├── observations/rgb_images/camera_{top,left,right} (T,) object  # encoded image bytes
├── puppet/end_effector                             (T, 6)
├── puppet/joint_position                           (T, 8)

end_effector (6d): [x,y,z,r,p,y]
joint_position (8d): [base link, ..., end_effector link, gripper]

The output matches what ``seeker.dataset.mimicgen_dataset.MimicGenDataset``
expects under ``cache_dir``:

<cache_dir>/
  images.lmdb
  meta.json
  build_done.flag
  task_instruction.npy
  lowdim/
    robot0_eef_pos.npy
    robot0_eef_rot.npy
    robot0_gripper_qpos.npy
    task_embedding.npy
    robot_id.npy
  action/
    absolute_action.npy
    relative_action.npy
    delta_action.npy
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import traceback

import cv2
import h5py
import lmdb
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from seeker.util.image_io import encode_rgb_to_jpg_bytes
from seeker.util.task_meta import (
    ROBOT_NAME_TO_ID,
    instruction_to_task_embedding,
    setup_task_embedding_cache,
)


DEFAULT_CAMERA_MAP = {
    "camera_top": "agentview_image",
    # TODO just skip the eye in hand cam, if this doesnt break code remove
    # "camera_left": "robot0_eye_in_hand_image", 
    }


@dataclass
class EpisodePayload:
    instruction: str
    images: Dict[str, np.ndarray]
    eef_pos: np.ndarray
    eef_rot: np.ndarray
    gripper_qpos: np.ndarray
    absolute_action: np.ndarray
    relative_action: np.ndarray
    delta_action: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        required=True,
        help="Dataset root that contains success_episodes/train/*/data/trajectory.hdf5.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="success_episodes/train/*/data/trajectory.hdf5",
        help="Glob relative to --data_dir used to collect raw episodes.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Optional limit after globbing episode files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output cache directory. Defaults to the same convention used by "
            "MimicGenDataset: <dataset_parent>/<dataset_parent_name>_lmdb."
        ),
    )
    parser.add_argument(
        "--camera-map",
        nargs="+",
        default=None,
        help=(
            "Raw-to-Seeker camera mapping as raw=target pairs. "
            "Default: camera_top=agentview_image "
            "camera_left=robot0_eye_in_hand_image" # TODO also remove eye in hand 
        ),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--lmdb-map-size-gb", type=int, default=100)
    parser.add_argument("--commit-every", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=16) # TODO: set a sensible horizon, I fear 16 is wayyy to small? more like 400+?
    parser.add_argument(
        "--orientation-format",
        type=str,
        choices=["euler_xyz", "rotvec", "quat_xyzw", "quat_wxyz"],
        help="Orientation format stored in puppet/end_effector[..., 3:]. Defaults to episode attr or euler_xyz.",
    )
    parser.add_argument(
        "--action_shift",
        type=int,
        default=1,
        help="Use pose at t + action_shift as the supervision target for step t.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="franka",
        choices=sorted(ROBOT_NAME_TO_ID.keys()),
        help="Robot id written into lowdim/robot_id.npy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite out_dir if it already exists.",
    )

    args = parser.parse_args()
    return args


def default_out_dir(input_path: Path) -> Path:
    parent = input_path.expanduser().resolve()#.parent
    return parent / f"{parent.name}_lmdb"


def resolve_input_files(args: argparse.Namespace) -> List[Path]:
    assert args.data_dir is not None, "data_dir is None"

    base_dir = Path(args.data_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {base_dir}")
    
    files = sorted(base_dir.glob(args.glob_pattern))
    print(f"... Found {len(files)} episodes")

    if args.num_episodes is not None:
        files = files[: int(args.num_episodes)]
        print(f"... Slicing the first {len(files)} episodes")

    if not files:
        raise FileNotFoundError(
            f"No files matched {args.glob_pattern!r} under {base_dir}"
        )
    
    return [p.resolve() for p in files]
    

def parse_camera_map(items: Sequence[str] | None) -> Dict[str, str]:
    if items is None:
        return dict(DEFAULT_CAMERA_MAP)

    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --camera_map item {item!r}; expected raw=target.")
        src, dst = item.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError(f"Invalid --camera_map item {item!r}.")
        mapping[src] = dst
    if not mapping:
        raise ValueError("camera_map cannot be empty")
    return mapping


def validate_and_encode_image(img: np.ndarray, quality: int, step_idx: int, cache_key: str) -> bytes:
    """Validate image and encode to JPEG with error checking."""
    try:
        # Check for empty image
        if img is None or img.size == 0:
            raise ValueError(f"Empty image at step {step_idx}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(img)) or np.any(np.isinf(img)):
            raise ValueError(f"Image contains NaN or Inf at step {step_idx}")
        
        # Check for valid dimensions
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Invalid image shape {img.shape} at step {step_idx}")
        
        # Check for valid values (0-255 for uint8)
        if img.dtype == np.uint8:
            if img.min() < 0 or img.max() > 255:
                print(f"    Warning: Image values outside 0-255 range: min={img.min()}, max={img.max()} at step {step_idx}")
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Encode
        jpg = encode_rgb_to_jpg_bytes(img, quality=quality)
        
        if len(jpg) == 0:
            raise ValueError(f"Encoded JPEG is empty at step {step_idx}")
        
        # Check if JPEG is valid (optional: try to decode it)
        test_decode = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if test_decode is None:
            raise ValueError(f"Encoded JPEG cannot be decoded at step {step_idx}")
        
        return jpg
        
    except Exception as e:
        print(f"[ERROR] encoding image {cache_key} at step {step_idx}: {e}")
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image range: [{img.min()}, {img.max()}]")
        raise


def ensure_empty_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            response = input(
                f"Output directory already exists: {out_dir}\n"
                "Do you want to overwrite it? [Y/n]: "
            ).strip().lower()

            if response not in {"","y", "yes"}:
                raise FileExistsError(
                    f"Output directory already exists: {out_dir}. Aborting."
                )

        shutil.rmtree(out_dir)

    (out_dir / "lowdim").mkdir(parents=True, exist_ok=True)
    (out_dir / "action").mkdir(parents=True, exist_ok=True)


def _sorted_demo_keys(group: h5py.Group) -> List[str]:
    keys = list(group.keys())

    def sort_key(name: str) -> Tuple[int, str]:
        suffix = name.split("_")[-1]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(keys, key=sort_key)


def _is_raw_episode(group: h5py.Group) -> bool:
    needed = ["observations", "puppet", "master"]
    return all(k in group for k in needed)


def iter_raw_episodes(h5_file: h5py.File) -> Iterable[Tuple[str, h5py.Group]]:
    if _is_raw_episode(h5_file):
        yield "demo_0", h5_file
        return

    if "data" not in h5_file:
        raise ValueError("HDF5 does not look like raw RoboMind data: missing raw root or /data.")

    data_group = h5_file["data"]
    found = False
    for name in _sorted_demo_keys(data_group):
        grp = data_group[name]
        if isinstance(grp, h5py.Group) and _is_raw_episode(grp):
            found = True
            yield name, grp

    if not found:
        raise ValueError("No raw RoboMind episodes found under /data.")


def dataset_scalar_string(dataset: h5py.Dataset) -> str:
    value = dataset[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8")
    return str(value)


def infer_orientation_format(group: h5py.Group, cli_value: str | None) -> str:
    if cli_value is not None:
        return cli_value

    for owner in (group, group.file):
        raw = owner.attrs.get("orientation_format")
        if raw is not None:
            if isinstance(raw, (bytes, np.bytes_)):
                return raw.decode("utf-8")
            return str(raw)
    return "euler_xyz"


def orientation_to_matrix(orientation: np.ndarray, orientation_format: str) -> np.ndarray:
    if orientation_format == "euler_xyz":
        return R.from_euler("xyz", orientation.astype(np.float64), degrees=False).as_matrix()
    if orientation_format == "rotvec":
        return R.from_rotvec(orientation.astype(np.float64)).as_matrix()
    if orientation_format == "quat_xyzw":
        return R.from_quat(orientation.astype(np.float64)).as_matrix()
    if orientation_format == "quat_wxyz":
        quat_xyzw = np.array(
            [orientation[1], orientation[2], orientation[3], orientation[0]],
            dtype=np.float64,
        )
        return R.from_quat(quat_xyzw).as_matrix()
    raise ValueError(f"Unsupported orientation_format: {orientation_format}")


def extract_gripper_qpos(joint_pos: np.ndarray) -> np.ndarray:
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be 2D, got shape {joint_pos.shape}")

    if joint_pos.shape[1] >= 9:
        return joint_pos[:, -2:].astype(np.float32)

    # RoboMind's 8-dim Panda trajectories normally store one scalar gripper value (I hope lol)
    g = joint_pos[:, -1:].astype(np.float32)
    return np.concatenate([g, -g], axis=1)


def gripper_action_scalar(gripper_qpos: np.ndarray) -> np.ndarray:
    return np.abs(gripper_qpos[:, 0] - gripper_qpos[:, 1]).astype(np.float32)[:, None]


def decode_image(img_bytes: bytes, image_size: int) -> np.ndarray:
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes from HDF5.")
    if image_size is not None and (
        img_bgr.shape[0] != image_size or img_bgr.shape[1] != image_size
    ):
        img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_images(
    group: h5py.Group,
    camera_map: Dict[str, str],
    image_size: int,
) -> Dict[str, np.ndarray]:
    rgb_group = group["observations/rgb_images"]
    out: Dict[str, np.ndarray] = {}
    for raw_name, target_name in camera_map.items():
        if raw_name not in rgb_group:
            raise KeyError(
                f"Missing raw camera {raw_name!r}. Available cameras: {list(rgb_group.keys())}"
            )
        ds = rgb_group[raw_name]

        frames = [decode_image(ds[i], image_size=image_size) for i in range(ds.shape[0])]
        out[target_name] = np.stack(frames, axis=0).astype(np.uint8)
    return out


def pose6_to_pos_rot(eef_pose: np.ndarray, orientation_format: str) -> Tuple[np.ndarray, np.ndarray]:
    eef_pose = np.asarray(eef_pose, dtype=np.float32)
    pos = eef_pose[:, :3].astype(np.float32)
    rot = np.stack(
        [
            orientation_to_matrix(eef_pose[i, 3:], orientation_format).reshape(9)
            for i in range(eef_pose.shape[0])
        ],
        axis=0,
    ).astype(np.float32)
    return pos, rot


def shift_targets(arr: np.ndarray, shift: int) -> np.ndarray:
    shift = int(shift)
    if shift < 0:
        raise ValueError("action_shift must be >= 0")
    if arr.shape[0] == 0:
        return arr.copy()
    idx = np.clip(np.arange(arr.shape[0]) + shift, 0, arr.shape[0] - 1)
    return arr[idx]


def make_absolute_action(target_pos: np.ndarray, target_rot: np.ndarray, target_grip: np.ndarray) -> np.ndarray:
    return np.concatenate([target_pos, target_rot, target_grip], axis=1).astype(np.float32)


def make_relative_action(
    curr_pos: np.ndarray,
    curr_rot: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    target_grip: np.ndarray,
) -> np.ndarray:
    curr_rot_m = curr_rot.reshape(-1, 3, 3)
    target_rot_m = target_rot.reshape(-1, 3, 3)
    dp_world = target_pos - curr_pos
    dp_body = np.einsum("tij,tj->ti", np.transpose(curr_rot_m, (0, 2, 1)), dp_world)
    d_rot = np.einsum("tij,tjk->tik", np.transpose(curr_rot_m, (0, 2, 1)), target_rot_m)
    return np.concatenate([dp_body, d_rot.reshape(-1, 9), target_grip], axis=1).astype(np.float32)


def to_se3_delta_action_chunk(
    eef_pos: np.ndarray,
    eef_rot: np.ndarray,
    absolute_action: np.ndarray,
    horizon: int,
) -> np.ndarray:
    out = []
    T = absolute_action.shape[0]
    for i in range(T):
        chunk = absolute_action[i : i + horizon]
        if chunk.shape[0] < horizon:
            pad = horizon - chunk.shape[0]
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)], axis=0)

        p0 = eef_pos[i]
        R0 = eef_rot[i].reshape(3, 3)
        pt = chunk[:, :3]
        Rt = chunk[:, 3:12].reshape(-1, 3, 3)
        gr = chunk[:, 12:]

        dp_world = pt - p0
        dp_body = np.einsum("ij,tj->ti", R0.T, dp_world)
        d_rot = np.einsum("ij,tjk->tik", R0.T, Rt).reshape(-1, 9)[:, :6]
        out.append(np.concatenate([dp_body, d_rot, gr], axis=1).reshape(1, -1))

    return np.concatenate(out, axis=0).astype(np.float32)


def build_episode_payload(
    group: h5py.Group,
    camera_map: Dict[str, str],
    image_size: int,
    orientation_format: str,
    action_shift: int,
    horizon: int,
) -> EpisodePayload:
    instruction = dataset_scalar_string(group["language_raw"]).strip()
    if not instruction:
        raise ValueError("Episode is missing language_raw instruction text.")

    images = load_images(group, camera_map=camera_map, image_size=image_size)
    joint_pos = np.asarray(group["puppet/joint_position"], dtype=np.float32)
    eef_pose = np.asarray(group["puppet/end_effector"], dtype=np.float32)

    if joint_pos.shape[0] != eef_pose.shape[0]:
        raise ValueError("puppet/joint_position and puppet/end_effector length mismatch.")

    eef_pos, eef_rot = pose6_to_pos_rot(eef_pose, orientation_format=orientation_format)
    gripper_qpos = extract_gripper_qpos(joint_pos)

    # We gotta shift all actions by 1 so that we have a[t]=q[t+1]
    target_pos = shift_targets(eef_pos, action_shift)
    target_rot = shift_targets(eef_rot, action_shift)
    target_grip = shift_targets(gripper_action_scalar(gripper_qpos), action_shift)

    absolute_action = make_absolute_action(target_pos, target_rot, target_grip)
    relative_action = make_relative_action(
        curr_pos=eef_pos,
        curr_rot=eef_rot,
        target_pos=target_pos,
        target_rot=target_rot,
        target_grip=target_grip,
    )
    delta_action = to_se3_delta_action_chunk(
        eef_pos=eef_pos,
        eef_rot=eef_rot,
        absolute_action=absolute_action,
        horizon=horizon,
    )

    return EpisodePayload(
        instruction=instruction,
        images=images,
        eef_pos=eef_pos,
        eef_rot=eef_rot,
        gripper_qpos=gripper_qpos,
        absolute_action=absolute_action,
        relative_action=relative_action,
        delta_action=delta_action,
    )




def main() -> None:
    args = parse_args()
    input_files = resolve_input_files(args)

    default_base = (
        input_files[0]
        if len(input_files) == 1
        else Path(args.data_dir if args.data_dir is not None else args.input).expanduser().resolve()
    )

    out_dir = default_out_dir(default_base) if args.out_dir is None else Path(args.out_dir).expanduser().resolve()

    camera_map = parse_camera_map(args.camera_map)

    ensure_empty_out_dir(out_dir, overwrite=bool(args.overwrite))

    robot_id = int(ROBOT_NAME_TO_ID[args.robot_name])

    setup_task_embedding_cache() # TODO: add embeddings for the new tasks from RoboMind???

    lmdb_path = out_dir / "images.lmdb"

    if lmdb_path.exists():
        print(f"... [WARNING] prev lmdb_path {lmdb_path} was not properly deleted")

    env = None
    try:
        env = lmdb.open(
            str(lmdb_path),
            map_size=int(args.lmdb_map_size_gb * (1024**3)),
            subdir=False,
            readonly=False,
            meminit=True,
            map_async=False, 
            max_dbs=1,
            lock=True,
        )

        # # Initialize LMDB with a dummy entry to avoid corruption
        # with env.begin(write=True) as txn:
        #     txn.put(b"__init__", b"initialized")

        txn = env.begin(write=True)
        put_count = 0

        episode_lengths: List[int] = []
        task_instructions: List[str] = []
        lowdim_eef_pos: List[np.ndarray] = []
        lowdim_eef_rot: List[np.ndarray] = []
        lowdim_gripper: List[np.ndarray] = []
        absolute_actions: List[np.ndarray] = []
        relative_actions: List[np.ndarray] = []
        delta_actions: List[np.ndarray] = []
        global_step = 0

        for input_path in tqdm(input_files, desc="Episodes", leave=False):            
            with h5py.File(input_path, "r") as h5_file:
                episodes = list(iter_raw_episodes(h5_file))
                if not episodes:
                    raise RuntimeError(f"No episodes found in input HDF5: {input_path}")
                

                task_embeddings = instruction_to_task_embedding(task_instructions).cpu().numpy().astype(np.float32)
                robot_ids = np.full((len(episode_lengths),), robot_id, dtype=np.int64)

                for _, group in episodes:

                    orientation_format = infer_orientation_format(group, args.orientation_format)
                    payload = build_episode_payload(
                        group=group,
                        camera_map=camera_map,
                        image_size=int(args.image_size),
                        orientation_format=orientation_format,
                        action_shift=int(args.action_shift),
                        horizon=int(args.horizon),
                    )

                    T = int(payload.eef_pos.shape[0])
                    episode_lengths.append(T)

                    task_instructions.append(payload.instruction)
                    lowdim_eef_pos.append(payload.eef_pos.astype(np.float32))
                    lowdim_eef_rot.append(payload.eef_rot.astype(np.float32))
                    lowdim_gripper.append(payload.gripper_qpos.astype(np.float32))
                    absolute_actions.append(payload.absolute_action.astype(np.float32))
                    relative_actions.append(payload.relative_action.astype(np.float32))
                    delta_actions.append(payload.delta_action.astype(np.float32))


                    for t in range(T):                        
                        for cache_key, frames in payload.images.items():
                            # TODO crop images? 
                            # img = frames[t]
                            # if img.shape[0] != 240 or img.shape[1] != 240:
                            #     img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
                            
                            lmdb_key = f"{cache_key}/{global_step:08d}".encode("ascii")
                            jpg = encode_rgb_to_jpg_bytes(frames[t], quality=int(args.jpeg_quality))
                            
                            txn.put(lmdb_key, jpg)
                            put_count += 1
                            if put_count % int(args.commit_every) == 0:
                                txn.commit()
                                txn = env.begin(write=True)

                        global_step += 1

        txn.put(b"__len__", str(global_step).encode("ascii"))
        txn.comit()
        env.sync()
        env.close()
        
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        raise e
    
    # Write arrays
    np.save(out_dir / "lowdim" / "robot0_eef_pos.npy", np.concatenate(lowdim_eef_pos, axis=0).astype(np.float32))
    np.save(out_dir / "lowdim" / "robot0_eef_rot.npy", np.concatenate(lowdim_eef_rot, axis=0).astype(np.float32))
    np.save(
        out_dir / "lowdim" / "robot0_gripper_qpos.npy",
        np.concatenate(lowdim_gripper, axis=0).astype(np.float32),
    )
    np.save(out_dir / "lowdim" / "task_embedding.npy", task_embeddings)
    np.save(out_dir / "lowdim" / "robot_id.npy", robot_ids)

    np.save(out_dir / "action" / "absolute_action.npy", np.concatenate(absolute_actions, axis=0).astype(np.float32))
    np.save(out_dir / "action" / "relative_action.npy", np.concatenate(relative_actions, axis=0).astype(np.float32))
    np.save(out_dir / "action" / "delta_action.npy", np.concatenate(delta_actions, axis=0).astype(np.float32))
    np.save(out_dir / "task_instruction.npy", np.asarray(task_instructions, dtype=object), allow_pickle=True)

    meta = {
        "source_files": [str(p) for p in input_files],
        "env_name": "",
        "camera_map": camera_map, # TODO remove?
        "rgb_keys": list(camera_map.values()),
        "lowdim_keys": ["robot0_eef_pos", "robot0_eef_rot", "robot0_gripper_qpos"],
        "episode_lengths": list(map(int, episode_lengths)),
        "n_demo": int(len(episode_lengths)),
        "n_samples": int(sum(episode_lengths)),
        "image_size": int(args.image_size),
        "horizon": int(args.horizon),
        "action_shift": int(args.action_shift),
        "robot_name": args.robot_name,
        "robot_id": robot_id,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(out_dir / "build_done.flag", "w", encoding="utf-8") as f:
        f.write("build completed\n")

    print(
        f"[built] out_dir={out_dir} demos={meta['n_demo']} samples={meta['n_samples']} rgb_keys={meta['rgb_keys']}"
    )


if __name__ == "__main__":
    main()
