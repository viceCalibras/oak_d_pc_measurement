#!/usr/bin/env python3
"""Module contains all the utility functions, mainly
thresholding and visualizations.
"""
from typing import Type
import open3d as o3d
import numpy as np
import plotly.graph_objs as go


def threshold_pc_distance(
    pc: Type[o3d.cpu.pybind.geometry.PointCloud],
    threshold_min: float,
    threshold_max: float,
) -> Type[o3d.cpu.pybind.geometry.PointCloud]:
    """Provides basic spatial filtering functionality - removes all the points that are
    outside a certain threshold interval.

    Args:
        pc: Input point cloud.
        threshold_min: Minimum distance threshold in mm.
        threshold_max: Maximum distance threshold in mm.

    Returns:
        Thresholded point cloud.
    """
    threshold_min = threshold_min / 1000  # OAK-D pc uses meters.
    threshold_max = threshold_max / 1000

    pc_array = np.asarray(pc.points)
    origin_distances = np.zeros(pc_array.shape)
    origin_distances = np.linalg.norm(origin_distances - pc_array, axis=1)

    mask = np.logical_and(
        origin_distances > threshold_min, origin_distances < threshold_max
    )

    pc_thresholded = o3d.geometry.PointCloud()
    pc_thresholded.points = o3d.utility.Vector3dVector(pc_array[mask])

    return pc_thresholded


def visualize_pc(
    pc: Type[o3d.cpu.pybind.geometry.PointCloud],
    point_size: float,
    window_name: str,
):
    """Custom visualizer function for the single point cloud.

    Args:
        pc: Input point cloud.
        point_size: Point size in the visualization.
        window_name: Title for the visualization window.

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=720, height=720, left=25, top=25)

    vis.add_geometry(pc)

    # Set rendering options.
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.point_size = point_size
    opt.light_on = True
    pc.paint_uniform_color([0, 0, 1])
    view = vis.get_view_control()
    view.rotate(250, 250)

    # Run the visualization.
    vis.run()


def visualize_two_pc(
    pc_1: Type[o3d.cpu.pybind.geometry.PointCloud],
    pc_2: Type[o3d.cpu.pybind.geometry.PointCloud],
    point_size: float,
    window_name: str,
    transformation: np.ndarray = None,
):
    """Custom visualizer function for 2 point clouds.

    Args:
        pc_1: First input point cloud.
        pc_2: Second input point cloud.
        point_size: Point size in the visualization.
        window_name: Title for the visualization window.
        transformation: Transformation array that is to be applied on the 2nd point cloud, if exists.
                        Used to visualize registration results.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=720, height=720, left=25, top=25)

    vis.add_geometry(pc_1)
    vis.add_geometry(pc_2)

    # Do the transformation, if required.
    if transformation is not None:
        pc_2.transform(transformation)

    # Set rendering options.
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.point_size = point_size
    opt.light_on = True
    pc_1.paint_uniform_color([1, 0.706, 0])
    pc_2.paint_uniform_color([0, 0.651, 0.929])
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True
    view = vis.get_view_control()
    view.rotate(250, 250)

    # Run the visualization.
    vis.run()
    vis.destroy_window()

def visualizer_pc(pc: Type[o3d.cpu.pybind.geometry.PointCloud]) -> Type[go.Figure]:
    """Creates a visualizer figure object, based on plotly.go.Scatter3d,
    that can be used to visualize point clouds. It is to be mostly used in the
    development environment as it can be rendered in the notebook.

    Args:
        pc: Input point cloud.

    Returns:
        Visualizer figure object. Use show() method to visualize.
    """
    pc = np.asarray(pc.points)
    vis = go.Figure(data =[go.Scatter3d(x = pc[:, 0],
                                   y = pc[:, 1],
                                   z = pc[:, 2],
                                   mode ='markers',           
                                    marker = dict(
                                     size = 1,
                                   ))])
    return vis
