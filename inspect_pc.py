#!/usr/bin/env python3
from typing import Type
import open3d as o3d
import numpy as np


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


def customDrawGeometry(model, data, point_size, window_name, transformation):
    """Custom visualizer function for the NAICP prototype.
    Args:
        model ([open3d geom object: pcd, triangle mesh, image): Referent pointset
        data ([open3d geom object: pcd, triangle mesh, image]): Pointset to be aligned
        point_size ([int]): Point size in visualization
        window_name ([string]): Title for the visualization window
        transformation ([array]): Transformation array for the data pointset
    """
    # Initialize the main Visualizer class
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=720, height=720, left=25, top=25)

    # Add the geometry to the scene and create corresponding shaders
    vis.add_geometry(model)
    vis.add_geometry(data)

    # Do the transformation, if required
    if transformation is not None:
        data.transform(transformation)

    # Rendering options
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.point_size = point_size
    opt.light_on = True
    # Paint two clouds
    model.paint_uniform_color([1, 0.706, 0])
    data.paint_uniform_color([0, 0.651, 0.929])
    # opt.mesh_color_option = Color
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True

    # Change the view
    view = vis.get_view_control()
    view.rotate(250, 250)

    # Run the visualization
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pc = o3d.io.read_point_cloud("./measurements/20230713-180927.pcd")
    visualize_pc(pc, 1.5, "Raw point cloud")

    pc_thresholded = threshold_pc_distance(pc, 2500, 3200)

    # visualize_pc(pc_thresholded, 1.5, "Filtered point cloud")
