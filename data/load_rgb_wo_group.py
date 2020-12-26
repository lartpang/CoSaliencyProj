import os
import random
import sys
from collections import defaultdict
from functools import partial

import albumentations as A
import cv2
import torch
from torch.nn import functional as F
from torch.utils import data

sys.path.append('..')
from configs import cosod_path
from data.utils import DataLoaderX, read_binary_array, read_color_array
from utils.misc import construct_print, set_seed_for_lib


def get_hw(in_size: dict) -> tuple:
    """
    从in_size中解析出h和w参数
    - 如果in_size仅有hw这个参数，那么就使用(hw,hw)
    - 如果指定了h和w，那么就是使用(h, w)
    """
    if new_size := in_size.get("hw", False):
        assert isinstance(new_size, int)
        new_size_tuple = (new_size, new_size)
    elif in_size.get("h", False) and in_size.get("w", False):
        assert isinstance(in_size["h"], int)
        assert isinstance(in_size["w"], int)
        new_size_tuple = (in_size["h"], in_size["w"])
    else:
        raise Exception(f"{in_size} error!")
    return new_size_tuple


def build_file_paths(data_info):
    image_root = data_info["image"]["path"]
    image_suffix = data_info["image"]["suffix"]
    mask_root = data_info["mask"]["path"]
    mask_suffix = data_info["mask"]["suffix"]

    data_paths = defaultdict(list)
    group_end_ids = []
    cur_group_end_id = 0
    for group_name in sorted(os.listdir(mask_root)):
        group_path = os.path.join(mask_root, group_name)

        file_name_list = sorted(os.listdir(group_path))
        cur_group_end_id += len(file_name_list)
        # 将当前图片组最后一张图片在整个数据集中的下标保存在 "group_end_ids" 中, 这部分信息是为 "Cosal_Sampler" 而准备的.
        group_end_ids.append(cur_group_end_id)

        for file_name in file_name_list:
            file_name = os.path.splitext(file_name)[0]
            data_paths["image"].append(os.path.join(image_root, group_name, file_name + image_suffix))
            data_paths["mask"].append(os.path.join(group_path, file_name + mask_suffix))
    return data_paths, group_end_ids


class TestDataset(data.Dataset):
    def __init__(self, root: dict, in_size: dict):
        """
        :param root: 这里的root是实际对应的数据字典
        :param in_size:
        """

        data_paths, group_end_ids = build_file_paths(data_info=root)
        construct_print(f"Loading data from: {root['root']}")
        self.total_image_path_list = data_paths["image"]
        self.total_mask_path_list = data_paths["mask"]
        self.indexes = group_end_ids

        self.new_size = get_hw(in_size)
        h, w = self.new_size
        self.img_transform = A.Compose(
            [
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )

    def __getitem__(self, index):
        curr_img_path = self.total_image_path_list[index]

        curr_img = read_color_array(curr_img_path)
        curr_size = curr_img.shape[:2]  # h, w

        transformed = self.img_transform(image=curr_img)
        curr_img = transformed["image"]
        curr_img_tensor = torch.from_numpy(curr_img).permute(2, 0, 1)

        curr_mask_path = self.total_mask_path_list[index]
        *_, group_name, mask_name = curr_mask_path.split(os.sep)

        return dict(
            image=curr_img_tensor,
            image_info=dict(
                ori_shape=curr_size,
                mask_path=curr_mask_path,
                mask_name=mask_name,
                group_name=group_name,
            ),
        )

    def __len__(self):
        return len(self.total_image_path_list)


class TrainDataset(data.Dataset):
    def __init__(self, root: list, in_size: dict):
        super(TrainDataset, self).__init__()
        self.scales = [1.0]
        if in_size.get("extra_scales"):
            self.scales += in_size["extra_scales"]

        self.total_image_path_list = []
        self.total_mask_path_list = []
        self.indexes = []
        for root_name, root_item in root:
            data_paths, group_end_ids = build_file_paths(data_info=root_item)
            construct_print(f"Loading data from {root_name}: {root_item['root']}")
            self.total_image_path_list += data_paths["image"]
            self.total_mask_path_list += data_paths["mask"]
            self.indexes += [i + len(self.indexes) for i in group_end_ids]

        self.new_size = get_hw(in_size)
        h, w = self.new_size
        self.joint_transform = A.Compose(
            [
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )

    def __getitem__(self, index):
        curr_img_path = self.total_image_path_list[index]
        curr_mask_path = self.total_mask_path_list[index]

        curr_img = read_color_array(curr_img_path)
        curr_mask = read_binary_array(curr_mask_path, thr=0)

        transformed = self.joint_transform(image=curr_img, mask=curr_mask)
        curr_img = transformed["image"]
        curr_mask = transformed["mask"]

        curr_img_tensor = torch.from_numpy(curr_img).permute(2, 0, 1)
        curr_mask_tensor = torch.from_numpy(curr_mask).unsqueeze(0)

        return dict(image=curr_img_tensor, mask=curr_mask_tensor)

    def __len__(self):
        return len(self.total_image_path_list)


def _worker_init_fn(worker_id, base_seed):
    set_seed_for_lib(base_seed + worker_id)


def _customized_collate_fn(batch, scales):
    assert isinstance(scales, (list, tuple))
    scale = random.choice(scales)

    recombined_data = defaultdict(list)
    for item_info in batch:
        for k, v in item_info.items():
            recombined_data[k].append(v)

    results = {}
    for k, v in recombined_data.items():
        stacked_tensor = torch.stack(v, dim=0)
        if k in ["mask"]:
            kwargs = dict(mode="nearest")
        else:
            kwargs = dict(mode="bilinear", align_corners=False)
        if float(torch.__version__[:3]) >= 1.6:
            kwargs["recompute_scale_factor"] = False
        results[k] = F.interpolate(stacked_tensor, scale_factor=scale, **kwargs)
    return results


def create_loader(
        data_path,
        in_size,
        batch_size,
        num_workers=4,
        base_seed=0,
        training=False,
        shuffle=False,
        use_mstrain=False,
        get_length=False,
        pin_memory=True,
):
    if training:
        dataset_obj = TrainDataset(root=data_path, in_size=in_size)
    else:
        dataset_obj = TestDataset(root=data_path, in_size=in_size)

    cus_collate_fn = partial(_customized_collate_fn, scales=dataset_obj.scales) if use_mstrain else None
    cus_worker_init_fn = partial(_worker_init_fn, base_seed=base_seed)
    loader = DataLoaderX(
        dataset=dataset_obj,
        collate_fn=cus_collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=cus_worker_init_fn,
    )
    if get_length:
        length_of_dataset = len(dataset_obj)
        return loader, length_of_dataset
    return loader


if __name__ == '__main__':
    dataset = TrainDataset(root=[('dsa', cosod_path.COCO9213)], in_size=dict(h=384, w=384, extra_scales=[0.5, 1.5]))
    cosal_sampler = CoSAL_Sampler(idxes=dataset.indexes, shuffle=False, batch_size=None, drop_last=True,
                                  fill_batch=True)
    loader = DataLoaderX(
        dataset=dataset,
        # collate_fn=partial(_customized_collate_fn, scales=dataset.scales),
        batch_sampler=cosal_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=partial(_worker_init_fn, base_seed=0),
    )
    print(cosal_sampler.batches_idxes)
    for i, batch in enumerate(loader):
        print(batch['image'].shape)
        if i == 10:
            break
