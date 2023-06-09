#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:27:11 2023

@author: peng
"""

import copy
import os
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
import cv2
import yaml
# from src.configs import Camera_cfg, CameraType

RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


CameraType = edict()
CameraType.realsense_d435i = 0
CameraType.realsense_l515 = 0

Camera_cfg = edict()
Camera_cfg.camera_type = 0  # 0: realsense d435; 1: realsense l515
Camera_cfg.realsense_d435i = edict()
Camera_cfg.realsense_d435i.image_size = (1280, 720)
Camera_cfg.realsense_d435i.fx = 920.451904296875
Camera_cfg.realsense_d435i.fy = 920.451904296875
Camera_cfg.realsense_d435i.cx = 631.1873779296875
Camera_cfg.realsense_d435i.cy = 370.04132080078125
Camera_cfg.realsense_d435i.k1 = 0
Camera_cfg.realsense_d435i.k2 = 0
Camera_cfg.realsense_d435i.p1 = 0
Camera_cfg.realsense_d435i.p2 = 0
Camera_cfg.realsense_d435i.k3 = 0
Camera_cfg.realsense_d435i.camera_height = 0.35  # unit m
Camera_cfg.realsense_d435i.camera_x_offset = 0.05  # unit m, distance between the origins of Lidar and camera

Camera_cfg.realsense_l515 = edict()

class ImageVisualization:
    def __init__(self, camera_type):
        if camera_type == CameraType.realsense_d435i:
            self.cfg = Camera_cfg.realsense_d435i
        elif camera_type == CameraType.realsense_l515:
            self.cfg = Camera_cfg.realsense_l515
        else:
            raise Exception("camera type is not defined")

        self.fig, self.ax = plt.subplots()
        self.intrinsic_matrix = self._get_intrinsic_matrix()
        self.dist_coeffs = np.array([self.cfg.k1, self.cfg.k2, self.cfg.p1, self.cfg.p2, self.cfg.k3, 0.0, 0.0, 0.0])
        self.point_color = RED
        self.edge_color = BLUE

        self.point_threshold = 1

    def _get_intrinsic_matrix(self):
        return np.array([[self.cfg.fx, 0.0, self.cfg.cx], [0.0, self.cfg.fy, self.cfg.cy], [0.0, 0.0, 1.0]])

    def _project_points(self, xy: np.ndarray):
        """
        Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

        Args:
            xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        Returns:
            uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
        """
        batch_size, horizon, _ = xy.shape

        # create 3D coordinates with the camera positioned at the given height
        xyz = np.concatenate([xy, -self.cfg.camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1)

        # create dummy rotation and translation vectors
        rvec = tvec = (0, 0, 0)

        xyz[..., 0] += self.cfg.camera_x_offset
        xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
        uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec,
                                  self.intrinsic_matrix, self.dist_coeffs)
        uv = uv.reshape(batch_size, horizon, 2)

        return uv

    def _get_pos_pixels(self, points: np.ndarray, clip: Optional[bool] = True):
        """
        Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
        Args:
            points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        Returns:
            pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
        """
        if len(points.shape) == 2:
            pts = points[np.newaxis]
        pts[:, :, 0] = np.clip(pts[:, :, 0], a_min=self.point_threshold, a_max=np.inf)
        pixels = self._project_points(pts)[0]
        pixels[:, 0] = self.cfg.image_size[0] - pixels[:, 0]
        
        pixels = np.array([[np.clip(p[0], 0, self.cfg.image_size[0]),
                                np.clip(p[1], 0, self.cfg.image_size[1])] for p in pixels])
        
        return pixels

    def plot_points_on_image(self, img: np.ndarray, list_points: list, index: int):
        """
        Plot trajectories and points on an image.
        If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.

        Args:
            ax: matplotlib axis
            img: image to plot
            list_points: list of points, each point is a numpy array of shape (2,)
            point_colors: list of colors for points
        """
        img = copy.deepcopy(img)
        img[:, :, [0, 2]] = img[:, :, [2, 0]]
        self.ax.imshow(img)
        point = list_points[:, :2]
        pt_pixels = self._get_pos_pixels(point, clip=True)
        self.ax.plot(
            pt_pixels[:, 0],
            pt_pixels[:, 1]+40,
            color=self.point_color,
            marker="o",
            markersize=10.0,
        )

        # Draw lines connecting the pt_pixels
        if len(pt_pixels) > 1:
            self.ax.plot(
                pt_pixels[:, 0],
                pt_pixels[:, 1]+40,
                color=self.edge_color,
                linewidth=2.0,
                linestyle="-",
            )

        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.set_xlim((0.5, self.cfg.image_size[0] - 0.5))
        self.ax.set_ylim((self.cfg.image_size[1] - 0.5, 0.5))



if __name__ == "__main__":
    
    for i in range(28):
        data = np.load('/home/peng/Downloads/test_data-20230530T140304Z-001/test_data/'+str(i)+'_0.pkl', allow_pickle=True)
        paths = data['all_paths']
        image = data['camera'][0]
        visual = ImageVisualization(0)
        # fig, ax = plt.subplots()
        num = len(paths)
    
        for j in range(num):
            paths[j][0]=np.array([0,0])
            # plot_points_on_image(ax, image,  paths[j], [RED, GREEN])
        visual.plot_points_on_image(image, paths[0], 0)
        visual.plot_points_on_image(image, paths[1], 0)  ## you can draw all path[i] on the same image (in the same ax) with different colors [point_color, line_color]
        print("test")