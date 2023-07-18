#!/usr/bin/env python3
"""Source: https://github.com/luxonis/depthai/blob/main/depthai_helpers/projector_3d.py
"""
from typing import List
from typing import Type
import time
import numpy as np
import open3d as o3d


class PointCloudGenerator:
    """Generates point cloud from the camera images. Provides visualization
    and point cloud (measuement) capture functionalities.
    Point clouds are generated from the RGBD images (RGB + depth), taking
    camera intrinsics into account.
    """

    def __init__(self, intrinsic_matrix: List[List], width: int, height: int):
        """
        Args:
            intrinsic_matrix : Camera intrinsics matrix.
            width: Image width, px.
            height: Image height, px.
        """
        self.depth_map = None
        self.rgb = None
        self.pc = None

        # Instantiate relevant open3d objects.
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic_matrix[0][0],
            intrinsic_matrix[1][1],
            intrinsic_matrix[0][2],
            intrinsic_matrix[1][2],
        )
        # self.vis = o3d.visualization.Visualizer()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.isstarted = False

        # Register key callbacks.
        self.vis.register_key_callback(ord("Q"), self._close_window)
        self.vis.register_key_callback(ord("C"), self._capture_pc)

    def _close_window(self, vis):
        """Stops the visualization."""
        vis.destroy_window()

    def _capture_pc(self, vis):
        """Captures a point cloud (measurement)."""
        timestr = time.strftime("%Y%m%d-%H%M%S")
        vis.capture_depth_point_cloud(f"./measurements/{timestr}.pcd")
        print(
            f"Point cloud measurement saved at: measurements/{timestr}.pcd",
        )

    def rgbd_to_pc(
        self, depth_map: np.ndarray, rgb: np.ndarray
    ) -> Type[o3d.cpu.pybind.geometry.PointCloud]:
        """Convert RGBD image to point cloud.
        Function first creates RGBD image from the RGB image and the depth map.
        Point cloud is created next, wrapping Open3D functionalities.

        Args:
            depth_map: Input depth map.
            rgb: Input RGB frame.

        Returns:
           Point cloud.
        """
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        is_rgb = False
        if len(self.rgb.shape) == 3:
            is_rgb = True

        if is_rgb:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
            )
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d
            )
        if self.pc is None:
            self.pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.pinhole_camera_intrinsic
            )
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.pinhole_camera_intrinsic
            )
            self.pc.points = pcd.points
            self.pc.colors = pcd.colors

        return self.pc

    def render_pc(self):
        """Renders projected point cloud."""
        if not self.isstarted:
            self.vis.add_geometry(self.pc)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pc)
            self.vis.poll_events()
            self.vis.update_renderer()
