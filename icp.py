#!/usr/bin/env python3
"""Registers all measurements into one output point cloud.
Script uses basic pairwise registration, appending final 
point cloud each time.
"""
import pathlib
import open3d as o3d
import numpy as np

from utils import threshold_pc_distance
from utils import visualize_pc

MEASUREMENTS_DIR = "./measurements/pumpkin/"

if __name__ == "__main__":
    pc_source = None
    for file in pathlib.Path(MEASUREMENTS_DIR).iterdir():
        if file.suffix == ".pcd":
            if pc_source is None:
                print("Loading first pc: ", file.name)
                pc_source = o3d.io.read_point_cloud(str(file))
                pc_source = threshold_pc_distance(pc_source, 250, 750)
            else:
                print("Loading target pc: ", file.name)
                pc_target = o3d.io.read_point_cloud(str(file))
                pc_target = threshold_pc_distance(pc_target, 250, 750)

                translation_vector = pc_source.get_center() - pc_target.get_center()
                pc_target.translate(translation_vector)

                # Estimate pc normals.
                radius_normal = 0.005  # 5 cm.
                pc_source.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_normal, max_nn=30
                    )
                )
                pc_target.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_normal, max_nn=30
                    )
                )

                print("Performing registration.")
                loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
                result = o3d.pipelines.registration.registration_icp(
                    pc_source,
                    pc_target,
                    0.01,
                    np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(
                        loss
                    ),
                )

                pc_target.transform(result.transformation)
                pc_source += pc_target

    # Prepare for visualization.
    pc_final = pc_source
    pc_final.voxel_down_sample(0.05)
    visualize_pc(pc_final, 1.0, "Measurement")

    # Do a surface reconstruction.
    # radius_normal = 0.005  # 5 cm.
    # pc_final.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=radius_normal, max_nn=30
    #     )
    # )
    # with o3d.utility.VerbosityContextManager(
    #     o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pc_final, depth=9)
        
    # print(mesh)
        
    # o3d.visualization.draw_geometries([mesh])

