#!/usr/bin/env python3
"""Registers all measurements into one output point cloud.
Script uses multiway registration via pose graph optimization.

Source: http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html
"""
from typing import Union
from typing import Type
from typing import List
from typing import Tuple
import pathlib
import open3d as o3d
import numpy as np

from utils import threshold_pc_distance
from utils import visualize_pc

MAX_CORRESPONDANCE_DISTANCE_COARSE = 0.3
MAX_CORRESPONDANCE_DISTANCE_FINE = 0.03

MEASUREMENTS_DIR = "./measurements/pumpkin/"

def load_point_clouds(measurements_dir: Union[pathlib.Path, str]) -> List:
    """Loads point clouds from the input directory into a single list.

    Args:
        measurements_dir: Directory containing input point clouds.

    Returns:
        List of point clouds.
    """
    pcs = []
    for file in pathlib.Path(measurements_dir).iterdir():
        if file.suffix == ".pcd":
            print("Loading pc: ", file.name)
            pc = o3d.io.read_point_cloud(str(file))
            pc = threshold_pc_distance(pc, 250, 750)
            radius_normal = 0.005  # 5 mm.
            pc.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_normal, max_nn=30
                    )
                )
            pcs.append(pc)

    return pcs


def pairwise_registration(source: Type[o3d.cpu.pybind.geometry.PointCloud],
                          target: Type[o3d.cpu.pybind.geometry.PointCloud], 
                          max_correspondence_distance_coarse: float, 
                          max_correspondence_distance_fine: float) -> Tuple[np.ndarray, np.ndarray]:
    """Executes single ICP registration on source and target point cloud.
    Registration consists of coarse and fine step.

    Args:
        source: Source input point cloud.
        target: Target input point cloud.
        max_correspondence_distance_coarse: Coarse max correspondance distance - ICP threshold.
        max_correspondence_distance_fine: Fine max correspondance distance - ICP threshold.

    Returns:
        Arrays containing resulting transformation and its meta data.
    """
    print("Executing pairwise registration.")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    
    return transformation_icp, information_icp


def full_registration(pcs: List[Type[o3d.cpu.pybind.geometry.PointCloud]],
                      max_correspondence_distance_coarse: float,
                      max_correspondence_distance_fine: float) -> o3d.cpu.pybind.pipelines.registration.PoseGraph:
    """Performs full registration loop, building pose graph
    along the way.

    Source: http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html#Pose-graph

    Args:
        pcs: Input point clouds.
        max_correspondence_distance_coarse: Coarse max correspondance distance - ICP threshold.
        max_correspondence_distance_fine: Fine max correspondance distance - ICP threshold.

    Returns:
        Resulting pose graph.
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcs = len(pcs)
    for source_id in range(n_pcs):
        for target_id in range(source_id + 1, n_pcs):
            transformation_icp, information_icp = pairwise_registration(
                pcs[source_id], pcs[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            
            print("Building up the pose graph.")
            if target_id == source_id + 1:  # Odometry case.
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # Loop closure case.
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

if __name__ == "__main__":
    pcs = load_point_clouds(MEASUREMENTS_DIR)
    # Perform pairwise registration & build up the pose graph.
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcs, MAX_CORRESPONDANCE_DISTANCE_COARSE, 
                                       MAX_CORRESPONDANCE_DISTANCE_FINE)
        
    print("Optimizing PoseGraph.")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=MAX_CORRESPONDANCE_DISTANCE_FINE,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    # Combine the point cloud.    
    pc_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcs)):
        pcs[point_id].transform(pose_graph.nodes[point_id].pose)
        pc_combined += pcs[point_id]

    visualize_pc(pc_combined, 1.0, "Measurement")
