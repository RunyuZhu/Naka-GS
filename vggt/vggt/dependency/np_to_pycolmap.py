# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pycolmap
from .projection import project_3D_points_np


def _use_modern_pycolmap_api():
    return hasattr(pycolmap, "Point2DList") and hasattr(pycolmap.Image(), "image_id")


def _build_point2d(xy, point3D_id):
    return pycolmap.Point2D(xy, point3D_id)


def _build_point2d_list(points2D_list):
    if hasattr(pycolmap, "Point2DList"):
        point2d_container = pycolmap.Point2DList()
        for point2d in points2D_list:
            point2d_container.append(point2d)
        return point2d_container

    if hasattr(pycolmap, "ListPoint2D"):
        return pycolmap.ListPoint2D(points2D_list)

    return points2D_list


def _ensure_rig(reconstruction, camera_id):
    if not _use_modern_pycolmap_api():
        return

    if reconstruction.exists_rig(camera_id):
        return

    rig = pycolmap.Rig(rig_id=camera_id)
    sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=camera_id)
    rig.add_ref_sensor(sensor)
    reconstruction.add_rig(rig)


def _add_image_to_reconstruction(
    reconstruction,
    image_id,
    image_name,
    camera_id,
    cam_from_world,
    points2D_list,
    registered=True,
):
    points2D = _build_point2d_list(points2D_list)

    if _use_modern_pycolmap_api():
        _ensure_rig(reconstruction, camera_id)

        image = pycolmap.Image(name=image_name, camera_id=camera_id, image_id=image_id)
        image.points2D = points2D
        image.frame_id = image_id

        frame = pycolmap.Frame(frame_id=image_id, rig_id=camera_id)
        reconstruction.add_frame(frame)
        reconstruction.frames[image_id].set_cam_from_world(camera_id, cam_from_world)
        reconstruction.frames[image_id].add_data_id(image.data_id)
        reconstruction.add_image(image)

        if registered:
            reconstruction.register_frame(image_id)
        return

    image = pycolmap.Image(id=image_id, name=image_name, camera_id=camera_id, cam_from_world=cam_from_world)
    image.points2D = points2D
    image.registered = registered
    reconstruction.add_image(image)


def _get_image_cam_from_world(pyimage):
    cam_from_world = getattr(pyimage, "cam_from_world", None)
    return cam_from_world() if callable(cam_from_world) else cam_from_world


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    if masks.sum(1).min() < min_inlier_per_frame:
        print(f"Not enough inliers per frame, skip BA.")
        return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    valid_point3D_ids = []
    for vidx in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        point3D_id = reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)
        valid_point3D_ids.append(point3D_id)

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        points2D_list = []

        point2D_idx = 0

        for point3D_id, original_track_idx in zip(valid_point3D_ids, valid_idx):

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(_build_point2d(point2D_xy, point3D_id))

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(fidx + 1, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        _add_image_to_reconstruction(
            reconstruction,
            image_id=fidx + 1,
            image_name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
            points2D_list=points2D_list,
            registered=True,
        )

    return reconstruction, valid_mask


def pycolmap_to_batch_np_matrix(reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"):
    """
    Convert a PyCOLMAP Reconstruction Object to batched NumPy arrays.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object from PyCOLMAP.
        device (str): Ignored in NumPy version (kept for API compatibility).
        camera_type (str): The type of camera model used (default: "SIMPLE_PINHOLE").

    Returns:
        tuple: A tuple containing points3D, extrinsics, intrinsics, and optionally extra_params.
    """

    point3D_ids = sorted(reconstruction.point3D_ids())
    max_points3D_id = max(point3D_ids)
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in point3D_ids:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for image_id in sorted(reconstruction.images):
        # Extract and append extrinsics
        pyimg = reconstruction.images[image_id]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = _get_image_cam_from_world(pyimg).matrix()
        extrinsics.append(matrix)

        # Extract and append intrinsics
        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    # Convert lists to NumPy arrays instead of torch tensors
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.stack(extra_params)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params


########################################################


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    point3D_ids = []
    for vidx in range(P):
        point3D_id = reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])
        point3D_ids.append(point3D_id)

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_ids[point3D_batch_idx]
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(_build_point2d(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        _add_image_to_reconstruction(
            reconstruction,
            image_id=fidx + 1,
            image_name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
            points2D_list=points2D_list,
            registered=True,
        )

    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri
