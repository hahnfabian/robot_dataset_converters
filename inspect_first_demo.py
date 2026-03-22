import h5py
import sys
import numpy as np

def write_demo_to_txt(file_path, demo_name="demo_0", output_file="demo_0_preview.txt", max_entries=5):
    with h5py.File(file_path, "r") as f:
        demo_group = f[f"data/{demo_name}"]
        with open(output_file, "w") as out:
            out.write(f"Preview of {demo_name} in {file_path}\n\n")
            
            for name, dataset in demo_group.items():
                if isinstance(dataset, h5py.Dataset):
                    out.write(f"Dataset: {name}  shape={dataset.shape}  dtype={dataset.dtype}\n")
                    # Limit output to first few entries
                    data_preview = dataset[tuple(slice(0, min(dim, max_entries)) for dim in dataset.shape)]
                    out.write(f"{data_preview}\n\n")
                elif isinstance(dataset, h5py.Group):
                    out.write(f"Group: {name}/\n")
                    for subname, subdataset in dataset.items():
                        if isinstance(subdataset, h5py.Dataset):
                            out.write(f"  Dataset: {subname}  shape={subdataset.shape}  dtype={subdataset.dtype}\n")
                            data_preview = subdataset[tuple(slice(0, min(dim, max_entries)) for dim in subdataset.shape)]
                            out.write(f"  {data_preview}\n\n")
    print(f"Preview written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python export_demo_preview.py <file.h5>")
        sys.exit(1)
    
    write_demo_to_txt(sys.argv[1])