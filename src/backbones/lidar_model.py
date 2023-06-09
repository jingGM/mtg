import time
from torch import nn
import torch
import warnings


class LidarImageModel(nn.Module):
    def __init__(self, input_channel=3, lidar_out_dim=512, norm_layer=True):
        super(LidarImageModel, self).__init__()
        if norm_layer:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([8, 12, 910]),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([16, 8, 453]),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([32, 6, 226]),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 2)), nn.ELU(), nn.LayerNorm([32, 2, 112]),
                nn.Flatten(), nn.Linear(7168, 2048), nn.LeakyReLU(0.2), nn.LayerNorm([2048]),
                nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.LayerNorm([1024]),
                nn.Linear(1024, lidar_out_dim), nn.ELU(), nn.LayerNorm([lidar_out_dim]),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 2)), nn.ELU(),
                nn.Flatten(), nn.Linear(7168, 2048), nn.LeakyReLU(0.2),
                nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, lidar_out_dim), nn.ELU()
            )

    def forward(self, image):
        output = self.conv(image)
        return output