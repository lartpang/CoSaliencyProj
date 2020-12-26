# -*- coding: utf-8 -*-
# @Time    : 2020/7/4
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import functools
from collections import defaultdict
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from metrics.metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from utils.misc import check_mkdir, construct_print


class MetricRecorder(object):
    def __init__(self):
        self.mae = MAE()
        self.fm = Fmeasure()
        self.sm = Smeasure()
        self.em = Emeasure()
        self.wfm = WeightedFmeasure()

    def update(self, pre: np.ndarray, gt: np.ndarray, gt_path: str):
        assert pre.shape == gt.shape, gt_path
        assert pre.dtype == np.uint8
        assert gt.dtype == np.uint8

        self.mae.step(pre, gt)
        self.sm.step(pre, gt)
        self.fm.step(pre, gt)
        self.em.step(pre, gt)
        self.wfm.step(pre, gt)

    def show(self, bit_num: int = 3) -> dict:
        fm_info = self.fm.get_results()
        fm = fm_info["fm"]
        pr = fm_info["pr"]
        wfm = self.wfm.get_results()["wfm"]
        sm = self.sm.get_results()["sm"]
        em = self.em.get_results()["em"]
        mae = self.mae.get_results()["mae"]
        results = {
            "em": em["curve"],
            "fm": fm["curve"],
            "Sm": sm,
            "wFm": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "adpFm": fm["adp"],
        }
        if isinstance(bit_num, int):
            results = {k: v.round(bit_num) for k, v in results.items()}
        return results


class GroupedMetricRecorder(object):
    def __init__(self):
        self.metric_recorders = {}

    def update(self, group_name, pre: np.ndarray, gt: np.ndarray, gt_path: str):
        if group_name not in self.metric_recorders:
            self.metric_recorders[group_name] = MetricRecorder()
        self.metric_recorders[group_name].update(pre, gt, gt_path)

    def show(self, bit_num: int = 3) -> dict:
        group_metrics = {k: v.show(bit_num=None) for k, v in self.metric_recorders.items()}
        results = self.mean_group_metrics(group_metric_recorder=group_metrics)
        results["meanFm"] = results["fm"].mean()
        results["maxFm"] = results["fm"].max()
        results["meanEm"] = results["em"].mean()
        results["maxEm"] = results["em"].max()
        del results["fm"]
        del results["em"]
        for k, v in results.items():
            temp_v = v.round(bit_num).tolist()
            if isinstance(temp_v, list):
                temp_v = temp_v[0]
            results[k] = temp_v
        return results

    @staticmethod
    def mean_group_metrics(group_metric_recorder: dict) -> dict:
        recorder = defaultdict(list)
        for group_name, metrics in group_metric_recorder.items():
            for metric_name, metric_array in metrics.items():
                recorder[metric_name].append(metric_array)
        results = {k: np.mean(np.vstack(v), axis=0) for k, v in recorder.items()}
        return results


class TBRecorder(object):
    def __init__(self, tb_path):
        check_mkdir(tb_path, delete_if_exists=True)

        self.tb = SummaryWriter(tb_path)

    def record_curve(self, name, data, curr_iter):
        if not isinstance(data, (tuple, list)):
            self.tb.add_scalar(f"data/{name}", data, curr_iter)
        else:
            for idx, data_item in enumerate(data):
                self.tb.add_scalar(f"data/{name}_{idx}", data_item[name], curr_iter)

    def record_image(self, name, data, curr_iter):
        data_grid = make_grid(data, nrow=data.size(0), padding=5)
        self.tb.add_image(name, data_grid, curr_iter)

    def record_histogram(self, name, data, curr_iter):
        self.tb.add_histogram(name, data, curr_iter)

    def close_tb(self):
        self.tb.close()


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        construct_print(f"a new epoch start: {start_time}")
        func(*args, **kwargs)
        construct_print(f"the time of the epoch: {datetime.now() - start_time}")

    return wrapper


def CustomizedTimer(cus_msg):
    def Timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            construct_print(f"{cus_msg} start: {start_time}")
            results = func(*args, **kwargs)
            construct_print(f"the time of {cus_msg}: {datetime.now() - start_time}")
            return results

        return wrapper

    return Timer
