
# Standard library imports for file and system operations
import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

# Third-party library imports for image processing, machine learning, and computer vision
import cv2
import open3d
import numpy as np
from rich.progress import track
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    write_cameras_text,
    write_images_text,
    read_points3D_binary,
    write_points3D_binary,
    write_points3D_text,
    Point3D,
)

# Custom module imports for shoe-splatter
import utils


def mask_pointcloud(recon_dir: Path, input_images_dir: Path, input_masks_dir: Path, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Mask the pointcloud using image masks and reconstruct 3D points with colors.

    This function processes a set of images and their corresponding masks to create
    a masked pointcloud. It uses camera information and 3D point data to reconstruct
    the pointcloud, applying depth constraints and error thresholds.

    Args:
        recon_dir (Path): Path to the directory containing reconstruction data (points3D, cameras, and images).
        input_images_dir (Path): Path to the directory containing input images.
        input_masks_dir (Path): Path to the directory containing input masks.
        verbose (bool): If True, displays progress information.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - points: An array of 3D point coordinates (N x 3).
            - colors: An array of corresponding RGB color values (N x 3).

    Raises:
        ValueError: If no valid pointcloud can be generated.

    Note:
        This function assumes specific file structures and naming conventions
        for the reconstruction data, images, and masks.
    """
    min_depth = 0.001
    max_depth = 10000
    max_repoj_err = 2.5
    min_n_visible = 2

    # Read pointcloud data and camera information
    ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    # Get camera intrinsic parameters
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height
    FX = cam_id_to_camera[CAMERA_ID].params[0]
    FY = cam_id_to_camera[CAMERA_ID].params[1]
    CX = cam_id_to_camera[CAMERA_ID].params[2]
    CY = cam_id_to_camera[CAMERA_ID].params[3]

    # Verbosity settings
    if verbose:
        iter_images = track(
            im_id_to_image.items(), total=len(im_id_to_image.items()), description="Masking ply from depth maps ..."
        )
    else:
        iter_images = iter(im_id_to_image.items())

    points = []
    colors = []

    # Iterate through images and masks to create depth maps and 3D pointcloud
    image_id_to_depth_path = {}
    for im_id, im_data in iter_images:
        out_name = str(im_data.name)
        mask_name = input_masks_dir / out_name
        image_name = input_images_dir / out_name

        # read mask and image
        mask = cv2.imread(str(mask_name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8) * 255
        image = cv2.imread(str(image_name))

        # get points in world coordinates
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])

        # get transformation from world to camera
        rotation = qvec2rotmat(im_data.qvec)
        world2cam = np.zeros((4,4))
        world2cam[:3, :3] = rotation
        world2cam[:3, 3] = im_data.tvec
        world2cam[3, 3] = 1.0
        cam2world = np.linalg.inv(world2cam)

        # project 3d points to camera coordinates
        z = (xyz_world @ world2cam[:3, :3].T + world2cam[:3, 3])[:,2]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array([im_data.xys[i] for i in range(len(im_data.xys)) if im_data.point3D_ids[i] != -1])
        
        # mask points within depth and w/h constraints and error thresholds
        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        # create depth map and apply mask
        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z
        valid_depth_mask = (mask == 255) & (depth > 0)
        depth[~valid_depth_mask] = 0

        # convert depth image to a list of xyz points in world space
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        u = (xs - CX) / FX
        v = (ys - CY) / FY
        depth_xyz_cam = depth[:, :, None] * np.stack([u, v, np.ones_like(u)], axis=-1)
        depth_xyz_cam = depth_xyz_cam[valid_depth_mask].reshape(-1, 3)
        depth_xyz_world = depth_xyz_cam @ cam2world[:3, :3].T + cam2world[:3, 3]

        # points
        points.append(depth_xyz_world)

        # colors
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        rgb_image = rgb_image[valid_depth_mask].reshape(-1, 3)
        colors.append(rgb_image)

    if len(points) > 0:
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
    else:
        raise ValueError("No valid pointcloud.")
    return points, colors


def main(input_path: Path) -> None:
    """
    Detect and save masks from an input image or directory.

    Args:
        input_path (Path): Path to the results workspace.
    
    Returns:
        None: No return value.
    """
    recon_dir = input_path / "colmap" / "sparse" / "0"
    new_recon_dir = input_path / "colmap" / "sparse" / "masked"
    input_images_dir = input_path / "images"
    input_masks_dir = input_path / "masks"
    output_path = input_path / "sparse_pc.ply"
    verbose= True

    utils.clear_and_create_directory(new_recon_dir)

    points, colors = mask_pointcloud(recon_dir, input_images_dir, input_masks_dir, verbose)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(str(output_path.resolve()), pcd)

    # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # inlier_cloud = cl.select_by_index(ind)
    # points = np.asarray(inlier_cloud.points)
    # colors = np.asarray(inlier_cloud.colors)
    # open3d.io.write_point_cloud(str(output_path.resolve()), inlier_cloud)

    colors = (colors * 255).astype(np.uint8)  # convert to 0-255 range
    points3D = {}
    for i in range(len(points)):
        points3D[i] = Point3D(id=i, xyz=points[i], rgb=colors[i],
                              error=0, image_ids=[], point2D_idxs=[])


    cameras = read_cameras_binary(recon_dir / "cameras.bin")
    images = read_images_binary(recon_dir / "images.bin")

    write_cameras_text(cameras, new_recon_dir / "cameras.txt")
    write_images_text(images, new_recon_dir / "images.txt")
    write_points3D_text(points3D, str((new_recon_dir / "points3D.txt").resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask the sparse pc from sfm with detections from grounded-sam.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the results workspace")
    args = parser.parse_args()

    main(args.input)