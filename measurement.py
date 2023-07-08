#!/usr/bin/env python3
from typing import Type
import open3d as o3d
import numpy as np


def threshold_pcl_distance(
    pcl: Type[o3d.open3d_pybind.geometry.PointCloud],
    threshold_min: float,
    threshold_max: float,
) -> Type[o3d.open3d_pybind.geometry.PointCloud]:
    """Provides basic spatial filtering functionality - removes all the points that are
    outside a certain threshold interval.

    Args:
        pcl: Input point cloud.
        threshold_min: Minimum distance threshold in mm.
        threshold_max: Maximum distance threshold in mm.

    Returns:
        Thresholded point cloud.
    """
    threshold_min = threshold_min / 1000  # OAK-D pcl uses meters.
    threshold_max = threshold_max / 1000

    pcl_array = np.asarray(pcl.points)
    origin_distances = np.zeros(pcl_array.shape)
    origin_distances = np.linalg.norm(origin_distances - pcl_array, axis=1)

    mask = np.logical_and(
        origin_distances > threshold_min, origin_distances < threshold_max
    )

    pcl_thresholded = o3d.geometry.PointCloud()
    pcl_thresholded.points = o3d.utility.Vector3dVector(pcl_array[mask])

    return pcl_thresholded


def visualize_pcl(
    pcl: Type[o3d.open3d_pybind.geometry.PointCloud],
    point_size: float,
    window_name: str,
):
    """Custom visualizer function for the single point cloud.

    Args:
        pcl: Input point cloud.
        point_size: Point size in the visualization.
        window_name: Title for the visualization window.

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=720, height=720, left=25, top=25)

    vis.add_geometry(pcl)

    # Set rendering options.
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.point_size = point_size
    opt.light_on = True
    pcl.paint_uniform_color([0, 0, 1])
    view = vis.get_view_control()
    view.rotate(250, 250)

    # Run the visualization.
    vis.run()


if __name__ == "__main__":
    pcl = o3d.io.read_point_cloud("./measurements/20230708-104324.pcd")
    visualize_pcl(pcl, 1.5, "Raw point cloud")

    pcl_thresholded = threshold_pcl_distance(pcl, 2500, 3200)

    visualize_pcl(pcl_thresholded, 1.5, "Filtered point cloud")
