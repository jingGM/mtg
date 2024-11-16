import time
from typing import Union, Callable, Tuple
import torch
from torch import nn, cuda, FloatTensor, LongTensor
from torch_cluster import knn
from src.utils.functions import get_device

UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]


def knn_indices_func_gpu(rep_pts: cuda.FloatTensor, pts: cuda.FloatTensor, k: int, sep: int) -> cuda.LongTensor:
    """
    @param rep_pts: representative points (N, rep_d, dim)
    @param pts: data points (N, pts_d, dim)
    @param k: nearest k number
    @return:
        indices of pts for each rep_pts, (N, rep_d, K)
    """
    device = get_device(device=rep_pts.device)
    n_batches, n_query, d = rep_pts.size()
    n_batches, n_data, d = pts.size()
    start_time = time.time()
    new_k = k * sep + 1
    data = pts.view(-1, d)
    queries = rep_pts.view(-1, d)
    data_length = torch.arange(n_batches).repeat_interleave(n_data).to(device)
    query_length = torch.arange(n_batches).repeat_interleave(n_query).to(device)
    knn_indices_gpu = knn(x=data, y=queries, k=new_k, batch_x=data_length,
                          batch_y=query_length)
    difference = torch.arange(n_batches) * n_data
    difference = difference.repeat_interleave(n_query * new_k)
    indices = knn_indices_gpu[1] - difference.to(device)
    all_indices = indices.view(n_batches, n_query, new_k)[:, :, 1::sep]
    print(time.time() - start_time, end=", ")
    return all_indices


class Dense(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, drop_rate: int = 0,
                 activation: Callable[[UFloatTensor], UFloatTensor] = nn.ELU()) -> None:
        """
        @param in_dim: Length of input featuers (last dimension).
        @param out_dim: Length of output features (last dimension).
        @param drop_rate: Drop rate to be applied after activation.
        @param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x: UFloatTensor) -> UFloatTensor:
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        if self.drop:
            x = self.drop(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]], with_bn: bool = True,
                 activation: Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x: UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x


class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 depth_multiplier: int = 1, with_bn: bool = True,
                 activation: Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups=in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias=not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x: UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x


def EndChannels(f):
    """ Class decorator to apply 2D convolution along end channels. """
    class WrappedLayer(nn.Module):
        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = self.f(x)
            x = x.permute(0, 2, 3, 1)
            return x
    return WrappedLayer()
