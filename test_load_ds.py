import tensorflow_datasets as tfds

def print_tree(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            print(prefix + k)
            print_tree(v, prefix + "  ")
    else:
        try:
            print(prefix + str(type(x)) + " " + str(getattr(x, "shape", "")))
        except:
            print(prefix + str(type(x)))

builder = tfds.builder("droid_100", data_dir="gs://gresearch/robotics")
ds = builder.as_dataset(split="train", shuffle_files=True)

# for episode in ds.take(1):

#     metadata = episode["episode_metadata"]
#     print(metadata)

#     steps = episode["steps"]

#     for step in steps.take(3):
#         print(step.keys())


for episode in ds.take(1):
    print_tree(episode)