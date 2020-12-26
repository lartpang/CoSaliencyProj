import os
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

object_list = []
dataset_root = ""
for video_name in tqdm(os.listdir(dataset_root), total=len(os.listdir(dataset_root))):
    video_path = os.path.join(dataset_root, video_name)
    first_frame_per_video_name = sorted(os.listdir(video_path))[0]
    first_frame_per_video_path = os.path.join(video_path, first_frame_per_video_name)
    first_frame_per_video = Image.open(first_frame_per_video_path)
    first_frame_per_video = np.asarray(first_frame_per_video)
    object_list.append(len(np.unique(first_frame_per_video)))

object_statistics = Counter(object_list)
print(object_statistics)
