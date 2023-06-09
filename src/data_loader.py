import copy
import os
import pickle
import random
import warnings
import cv2
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from random import shuffle

from src.configs import DataName, LidarMode


def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def registration_collate_fn_stack_mode(data_dicts):
    r"""Collate function for registration in stack mode.
    Args:
        data_dicts (List[Dict])
    Returns:
        collated_dict (Dict)
    """
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if key == DataName.all_paths:
                value = [torch.from_numpy(np.asarray(path)).to(torch.float) for path in value]
            else:
                value = torch.from_numpy(np.asarray(value)).to(torch.float)
            if key not in collated_dict:
                collated_dict[key] = []
            if key == DataName.path:
                if DataName.last_poses not in collated_dict:
                    collated_dict[DataName.last_poses] = [value[-1, :]]
                else:
                    collated_dict[DataName.last_poses].append(value[-1, :])
            collated_dict[key].append(value)

    for key, value in collated_dict.items():
        if key == DataName.path or key == DataName.scan or key == DataName.all_paths:
            pass
        else:
            collated_dict[key] = torch.stack(value, dim=0)

    return collated_dict


class GNDataset(Dataset):
    def __init__(self, data_file: str, data_name: str, train: bool, data_percentage=0.8, lidar_mode=LidarMode.image,
                 lidar_max_num=5210, lidar_vx_size=0.08, lidar_threshold=100., vel_num=1.2, w_eval=False,
                 use_local_map=False, local_map_threshold=200):
        with open(os.path.join(data_file, data_name), "rb") as input_file:
            dataset = pickle.load(input_file)
        self.root = data_file
        self.train = train
        self.w_eval = w_eval
        self.paths = dataset["root"][0].split("/")[-1]
        self.figures = dataset["root"][1].split("/")[-1]
        self.all_sample_indices = dataset["ids"]
        shuffle(self.all_sample_indices)

        select_split = int(len(self.all_sample_indices) * data_percentage)
        if self.train:
            self.data_indices = self.all_sample_indices[:select_split]
        else:
            self.data_indices = self.all_sample_indices[-select_split:]

        self.lidar_max_num = lidar_max_num
        self.lidar_vx_size = lidar_vx_size
        self.lidar_mode = lidar_mode
        self.lidar_threshold = lidar_threshold
        self.use_local_map = use_local_map
        self.vel_num = vel_num
        self.local_map_threshold = local_map_threshold

    def __len__(self):
        return len(self.data_indices)

    def _process_lidar(self, batched_pts):
        process_lidar = []
        for pts in batched_pts:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
            downpcd = pcd.voxel_down_sample(voxel_size=self.lidar_vx_size)
            downpcd = np.asarray(downpcd.points, dtype=np.float32)
            N = downpcd.shape[0]
            if N > self.lidar_max_num:
                selected_indices = random.sample(list(range(N)), self.lidar_max_num)
                process_lidar.append(downpcd[selected_indices])
            elif N < self.lidar_max_num:
                PN = downpcd.shape[0]
                if PN >= self.lidar_max_num:
                    selected_indices = random.sample(list(range(PN)), self.lidar_max_num)
                    process_lidar.append(downpcd[selected_indices])
                else:
                    warnings.warn("points number {} is smaller than required {}".format(PN, self.lidar_max_num))
                    process_lidar.append(np.concatenate((downpcd, np.zeros((self.lidar_max_num - PN, 3))), axis=0))
            else:
                process_lidar.append(downpcd)
        return np.array(process_lidar)

    def _process_local_map(self, local_map):
        cropped_local_map = local_map[int(local_map.shape[0] / 2 - self.local_map_threshold):
                                      int(local_map.shape[0] / 2 + self.local_map_threshold),
                            int(local_map.shape[1] / 2 - self.local_map_threshold):
                            int(local_map.shape[1] / 2 + self.local_map_threshold)]
        w = np.arange(cropped_local_map.shape[0])
        h = np.arange(cropped_local_map.shape[1])
        h_ex = np.tile(h, cropped_local_map.shape[0])
        w_ex = np.repeat(w, cropped_local_map.shape[1])
        mask = (w_ex % 2 != 0) | (h_ex % 2 != 0) | (
                np.linalg.norm(np.asarray([w_ex - cropped_local_map.shape[0] / 2.0,
                                           h_ex - cropped_local_map.shape[1] / 2.0]),
                               axis=0) > self.local_map_threshold)

        cropped_local_map[w_ex[mask], h_ex[mask]] = 0
        return cropped_local_map

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.paths, self.data_indices[index][0]), 'rb') as handle:
            metadata = pickle.load(handle)

        output_dict = {DataName.vel: np.asarray(metadata[DataName.vel])[:self.vel_num],
                       DataName.path: metadata[DataName.path]}
        if DataName.all_paths in metadata.keys():
            output_dict.update({DataName.all_paths: metadata[DataName.all_paths]})
        # local_map = cv2.imread(os.path.join(self.figures, self.data_indices[index][1]))
        # imu = metadata["imu"]
        # time = metadata["time"]
        # pose = metadata["pose"]
        # target = metadata["path"][-1]
        if self.lidar_mode == LidarMode.ptcnn:
            lidar_data = self._process_lidar(metadata[DataName.lidar])
        elif self.lidar_mode == LidarMode.image:
            if self.lidar_threshold is not None:
                lidar_data = np.asarray(metadata[DataName.lidar2d]) / self.lidar_threshold
            else:
                lidar_data = np.asarray(metadata[DataName.lidar2d])
        else:
            raise Exception("lidar model is not correct")
        output_dict.update({DataName.lidar: lidar_data, })

        local_map = self._process_local_map(metadata[DataName.local_map])
        output_dict.update({DataName.png: local_map})
        if self.w_eval or not self.train:
            output_dict.update({DataName.scan: metadata[DataName.scan],
                                DataName.local_map: np.asarray([int(self.data_indices[index][1][:-4])]), })
        return output_dict


def train_eval_data_loader(cfg):
    train_dataset = GNDataset(cfg.file, data_name=cfg.name, train=True, data_percentage=cfg.training_data_percentage,
                              lidar_mode=cfg.lidar_mode, lidar_threshold=cfg.lidar_threshold,
                              use_local_map=cfg.use_local_map, w_eval=cfg.w_eval,
                              vel_num=cfg.vel_num, local_map_threshold=cfg.local_map_threshold,
                              lidar_max_num=cfg.lidar_max_points, lidar_vx_size=cfg.lidar_downsample_vx_size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        sampler=None,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=False,
        drop_last=False,
    )

    eval_dataset = GNDataset(cfg.file, data_name=cfg.name, train=False, data_percentage=1 - cfg.training_data_percentage,
                             lidar_mode=cfg.lidar_mode, lidar_threshold=cfg.lidar_threshold,
                             use_local_map=cfg.use_local_map, w_eval=cfg.w_eval,
                             vel_num=cfg.vel_num, local_map_threshold=cfg.local_map_threshold,
                             lidar_max_num=cfg.lidar_max_points, lidar_vx_size=cfg.lidar_downsample_vx_size)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        sampler=None,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, eval_loader
