import h5py
import sys

def print_tree(name, obj):
    depth = name.count("/")
    indent = "│   " * depth
    node = name.split("/")[-1] if name else "/"

    if isinstance(obj, h5py.Group):
        print(f"{indent}├── {node}/")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}├── {node}  shape={obj.shape} dtype={obj.dtype}")

    # print attributes
    for k, v in obj.attrs.items():
        print(f"{indent}│   └── @{k}: {v}")

def explore_hdf5_tree(file_path):
    with h5py.File(file_path, "r") as f:
        print(f"/  ({file_path})")
        f.visititems(print_tree)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explore_hdf5_tree.py <file.h5>")
        sys.exit(1)

    explore_hdf5_tree(sys.argv[1])