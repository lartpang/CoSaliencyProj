import os

import torch
from tqdm import tqdm

from data.utils import read_binary_array
from utils.recorder import MetricRecorder

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(pred_root, mask_root):
    mask_name_list = sorted(os.listdir(mask_root))
    cal_total_seg_metrics = MetricRecorder()
    tqdm_iter = tqdm(enumerate(mask_name_list), total=len(mask_name_list), leave=False)
    for i, mask_name in tqdm_iter:
        tqdm_iter.set_description(f"te=>{i + 1} ")
        mask_array = read_binary_array(path=os.path.join(mask_root, mask_name))
        pred_array = read_binary_array(path=os.path.join(pred_root, mask_name))
        cal_total_seg_metrics.step(pred_array, mask_array, mask_name)
    fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results


def main():
    seg_results = test(pred_root="", mask_root="")
    print(seg_results)


if __name__ == "__main__":
    main()
