import numpy as np
from typing import Union, Optional
import cv2
import torch
import torch as th
from scipy.spatial.transform import Rotation


def get_device(device: Union[th.device, str] = "cuda") -> th.device:
    if isinstance(device, str):
        assert device == "cuda" or device == "cuda:0" or device == "cuda:1" or device == "cpu", \
            "device should only be 'cuda' or 'cpu' "
    device = th.device(device)
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")
    return device


def to_device(x, device):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_device(item, device) for item in x]
    elif isinstance(x, tuple):
        x = (to_device(item, device) for item in x)
    elif isinstance(x, dict):
        x = {key: to_device(value, device) for key, value in x.items()}
    elif isinstance(x, th.Tensor):
        x = x.to(device)
    return x


def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, th.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def pairwise_distance(x: th.Tensor, y: th.Tensor, normalized: bool = False, channel_first: bool = False,
                      app=False) -> th.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if app:
        return th.sum((x[:, :, None] - y[:, None]) ** 2, dim=-1)
    if channel_first:
        channel_dim = -2
        xy = th.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = th.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = th.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = th.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = th.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def index_select(data: th.Tensor, index: th.LongTensor, dim: int) -> th.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    data_shape = data.shape
    index_shape = index.shape
    if not data.is_contiguous():
        data = data.contiguous()
    if not index.is_contiguous():
        index = index.contiguous()
    th.cuda.empty_cache()
    output = data.index_select(dim, index.view(-1))
    if index.ndim > 1:
        output_shape = data_shape[:dim] + index_shape + data_shape[dim:][1:]
        output = output.view(*output_shape)

    return output


def apply_batch_transform(points: th.Tensor, transform: th.Tensor):
    r"""
    Points: B, N, 3
    transform: (B, 4, 4) or (B, 3, 4)
    """
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3].unsqueeze(1)
    points_shape = points.shape
    points = points.reshape(points_shape[0], -1, 3)
    points = th.matmul(points, rotation.transpose(-1, -2)) + translation
    points = points.reshape(*points_shape)
    return points


def single_transform(points: th.Tensor, rotation: th.Tensor, translation: th.Tensor):
    points_shape = points.shape
    points = points.view(-1, 3)
    points = th.matmul(points, rotation.transpose(-1, -2)) + translation
    points = points.reshape(*points_shape)
    return points

def apply_transform_np(points: np.ndarray, transform: np.ndarray):
    rotation = transform[:3, :3]  # (3, 3)
    translation = transform[None, :3, 3]  # (1, 3)
    points = np.matmul(points, rotation.transpose(-1, -2)) + translation
    return points

def apply_transform(points: th.Tensor, transform: th.Tensor, normals: Optional[th.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = th.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = th.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = th.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = th.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points


def apply_rotation(points: th.Tensor, rotation: th.Tensor, normals: Optional[th.Tensor] = None):
    r"""Rotate points and normals (optional) along the origin.

    Given a point cloud P(3, N), normals V(3, N) and a rotation matrix R, the output point cloud Q = RP, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), rotation is (3, 3), the output points are (*, 3).
       In this case, the rotation is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 3, 3), the output points are (B, N, 3).
       In this case, the rotation is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        rotation (Tensor): (3, 3) or (B, 3, 3)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if rotation.ndim == 2:
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = th.matmul(points, rotation.transpose(-1, -2))
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = th.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif rotation.ndim == 3 and points.ndim == 3:
        points = th.matmul(points, rotation.transpose(-1, -2))
        if normals is not None:
            normals = th.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and rotation{}.'.format(tuple(points.shape), tuple(rotation.shape))
        )
    if normals is not None:
        return points, normals
    else:
        return points


def get_rotation_translation_from_transform(transform):
    r"""Decompose transformation matrix into rotation matrix and translation vector.

    Args:
        transform (Tensor): (*, 4, 4)

    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def zyx_to_rotation(z, y, x) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.array(
        [np.arctan2(np.sin(z), np.cos(z)), np.arctan2(np.sin(y), np.cos(y)), np.arctan2(np.sin(x), np.cos(x))])
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def vector_to_transformation(vec):
    rotation = zyx_to_rotation(vec[-1], vec[-2], vec[-3])
    tanslation = np.array([[vec[0]], [vec[1]], [vec[2]]])
    transformation = np.concatenate((rotation, tanslation), axis=1)
    transformation = np.concatenate((transformation, np.array([[0, 0, 0, 1]])), axis=0)
    return transformation


def transformation_to_vector(transform):
    if type(transform) != np.ndarray:
        try:
            transform=transform.detach().cpu().numpy()
        except:
            transform=transform.cpu().numpy()
    if len(transform.shape) == 3:
        transform = transform[0]
    rz_o, ry_o, rx_o = rotation_to_zys(transform)
    t_o = transform[:3, 3]
    o_vec = np.array([t_o[0], t_o[1], t_o[2], rx_o, ry_o, rz_o])
    return o_vec


def rotation_to_zys(matrix):
    if type(matrix) != np.ndarray:
        matrix = matrix.cpu().numpy()
    z, y, x = Rotation.from_matrix(matrix[:3, :3]).as_euler(seq="zyx")
    return z, y, x


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def write_png(local_map=None, rgb_local_map=None, center=None, targets=None, paths=None, path=None,
              others=None, file="/home/jing/Documents/gn/global_nav/test/test_local_map.png"):
    dis = 2
    if rgb_local_map is not None:
        local_map_fig = rgb_local_map
    else:
        local_map_fig = np.repeat(local_map[:, :, np.newaxis], 3, axis=2) * 255
    if center is not None:
        assert center.shape[0] == 2 and len(center.shape) == 1, "path should be 2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(center + np.array([x, y]))
        all_points = np.stack(all_points).astype(int)
        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0
    if targets is not None and len(targets)>0:
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(targets + np.array([x, y]))
        if len(targets.shape) == 2:
            all_points = np.concatenate(all_points, axis=0).astype(int)
        else:
            all_points = np.stack(all_points, axis=0).astype(int)
        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0
    if others is not None:
        assert others.shape[1] == 2 and len(others.shape) == 2, "path should be Nx2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(others + np.array([x, y]))
        all_points = np.concatenate(all_points, axis=0).astype(int)
        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 255
    if path is not None:
        assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
        all_pts = np.asarray(path, dtype=int)
        all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                  all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                  all_pts), axis=0)
        local_map_fig[all_pts[:, 0], all_pts[:, 1], 0] = 0
        local_map_fig[all_pts[:, 0], all_pts[:, 1], 1] = 255
        local_map_fig[all_pts[:, 0], all_pts[:, 1], 2] = 255
    if paths is not None:
        for path in paths:
            if len(path) == 1 or np.any(path[0] == np.inf):
                continue
            path = np.asarray(path, dtype=int)
            assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
            all_pts = path
            all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                      all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                      all_pts), axis=0)
            local_map_fig[all_pts[:, 0], all_pts[:, 1], 0] = 0
            local_map_fig[all_pts[:, 0], all_pts[:, 1], 1] = 255
            local_map_fig[all_pts[:, 0], all_pts[:, 1], 2] = 255
    cv2.imwrite(file, local_map_fig)
    return local_map_fig