#!/usr/bin/env python3
import pathlib
import open3d as o3d
import numpy as np

from inspect_pc import threshold_pc_distance
from inspect_pc import visualize_pc

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
                radius_normal = 0.05  # 5 cm.
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

                print("Performing registration...")
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
    pc_source.voxel_down_sample(0.05)
    visualize_pc(pc_source, 1.0, "ICP")
