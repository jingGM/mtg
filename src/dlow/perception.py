import time
import warnings
import torch
from torch import nn
from src.backbones.lidar_model import  LidarImageModel
from src.configs import DataName, LidarMode


class Perception(nn.Module):
    def __init__(self, cfg):
        super(Perception, self).__init__()
        self.cfg = cfg

        self.lidar_model = LidarImageModel(input_channel=self.cfg.lidar_num, lidar_out_dim=self.cfg.lidar_out,
                                           norm_layer=self.cfg.lidar_norm_layer)

        self.vel_model = nn.Sequential(
            nn.Linear(self.cfg.vel_dim, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, self.cfg.vel_out), nn.LeakyReLU(0.2)
        )

    def forward(self, input_dict):
        lidar = input_dict[DataName.lidar]
        vel = input_dict[DataName.vel]

        if self.cfg.lidar_mode == LidarMode.ptcnn:
            B, K, N, D = lidar.size()
            lidar_fts_comb = self.lidar_model(lidar.view(-1, N, D))  # B*K x 512
            lidar_fts = lidar_fts_comb.view(B, K, -1)  # B x K x 512
            lidar_fts = torch.swapaxes(lidar_fts, axis0=-1, axis1=-2)  # B x 512 x k
            lidar_fts = self.lidar_combo(lidar_fts)  # B x 1024 x k
            lidar_fts = self.lidar_dense(lidar_fts.view(B, -1))  # B x 512
        elif self.cfg.lidar_mode == LidarMode.image:
            lidar_fts = self.lidar_model(lidar)  # B x 512
        else:
            raise Exception("lidar model is not correct")

        VB, VN, VD = vel.size()
        vel_fts = self.vel_model(vel.view(VB, -1))  # B x 256

        observation = torch.concat((lidar_fts, vel_fts), dim=1)  # B x 768
        return observation
