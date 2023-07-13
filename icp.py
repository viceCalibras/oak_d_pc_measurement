#!/usr/bin/env python3
from typing import Type
import open3d as o3d
import numpy as np

from inspect_pc import threshold_pc_distance
from inspect_pc import visualize_pc
from inspect_pc import customDrawGeometry

if __name__ == "__main__":
    pc_source = o3d.io.read_point_cloud("./measurements/lamp/20230713-180844.pcd")
    pc_target = o3d.io.read_point_cloud("./measurements/lamp/20230713-180853.pcd")

    # Threshold both pcs.
    pc_source = threshold_pc_distance(pc_source, 1500, 3000)
    pc_target = threshold_pc_distance(pc_target, 1500, 3000)

    translation_vector = pc_source.get_center() - pc_target.get_center()
    pc_target.translate(translation_vector)

    # Estimate pc normals.
    radius_normal = 0.05  # 5 cm.
    pc_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pc_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = 50  # Radius of the K-neighbourhood.
    max_nn_fpfh = 100  # Maximum number of neighbours that is to be searched.

    result = o3d.pipelines.registration.registration_icp(
        pc_source,
        pc_target,
        0.01,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    # N.B TransformationEstimationPointToPoint does not require normals!

    # visualize_pc(result, 1.5, "Measurement")
    customDrawGeometry(
        pc_source, pc_target, 1.5, "Global alignment", result.transformation
    )
