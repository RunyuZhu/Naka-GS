# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import os.path as osp
import random
from typing import Dict, List

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2, threshold_depth_map


class SevenScenesDataset(BaseDataset):
    """
    7Scenes dataset loader for VGGT training.

    Expected frame files in each sequence directory:
      - frame-XXXXXX.color.png
      - frame-XXXXXX.depth.png
      - frame-XXXXXX.pose.txt
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SEVENSCENES_DIR: str = "data/7Scenes",
        scenes: List[str] = None,
        min_num_images: int = 2,
        len_train: int = 100000,
        len_test: int = 10000,
        depth_scale: float = 1000.0,
        depth_max: float = 20.0,
        fx: float = 585.0,
        fy: float = 585.0,
        cx: float = 320.0,
        cy: float = 240.0,
        pose_is_c2w: bool = True,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.SEVENSCENES_DIR = SEVENSCENES_DIR
        self.min_num_images = min_num_images
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.pose_is_c2w = pose_is_c2w

        if split == "train":
            self.len_train = len_train
            split_filename = "TrainSplit.txt"
        elif split in ["val", "test"]:
            self.len_train = len_test
            split_filename = "TestSplit.txt"
        else:
            raise ValueError(f"Invalid split: {split}")

        if not osp.isdir(self.SEVENSCENES_DIR):
            raise FileNotFoundError(f"SEVENSCENES_DIR not found: {self.SEVENSCENES_DIR}")

        if scenes is None:
            scenes = sorted([d for d in os.listdir(self.SEVENSCENES_DIR) if osp.isdir(osp.join(self.SEVENSCENES_DIR, d))])
        elif isinstance(scenes, str):
            scenes = [s.strip() for s in scenes.split(",") if s.strip()]

        if self.debug:
            scenes = scenes[:1]

        self.sequence_list = self._build_sequence_list(scenes, split_filename)
        self.sequence_list_len = len(self.sequence_list)

        if self.sequence_list_len == 0:
            raise RuntimeError(f"No valid sequences found for split={split} in {self.SEVENSCENES_DIR}.")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: SevenScenes sequences: {self.sequence_list_len}")
        logging.info(f"{status}: SevenScenes dataset length: {len(self)}")

    def _build_sequence_list(self, scenes: List[str], split_filename: str) -> List[Dict]:
        seq_infos = []
        for scene in scenes:
            scene_dir = osp.join(self.SEVENSCENES_DIR, scene)
            split_path = osp.join(scene_dir, split_filename)
            if not osp.isfile(split_path):
                continue

            with open(split_path, "r") as f:
                split_lines = [line.strip() for line in f if line.strip()]

            for line in split_lines:
                # e.g. "sequence1" -> "seq-01"
                seq_id_str = line.replace("sequence", "").strip()
                if not seq_id_str.isdigit():
                    continue
                seq_dir = osp.join(scene_dir, f"seq-{int(seq_id_str):02d}")
                if not osp.isdir(seq_dir):
                    continue

                color_files = sorted(glob.glob(osp.join(seq_dir, "frame-*.color.png")))
                frame_basenames = [f[: -len(".color.png")] for f in color_files]
                if len(frame_basenames) < self.min_num_images:
                    continue

                seq_infos.append(
                    {
                        "scene": scene,
                        "seq_dir": seq_dir,
                        "seq_name": f"{scene}/{osp.basename(seq_dir)}",
                        "frame_basenames": frame_basenames,
                    }
                )
        return seq_infos

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_info = self.sequence_list[seq_index]
        else:
            seq_candidates = [s for s in self.sequence_list if s["seq_name"] == seq_name]
            if len(seq_candidates) == 0:
                raise ValueError(f"Sequence not found: {seq_name}")
            seq_info = seq_candidates[0]

        frame_basenames = seq_info["frame_basenames"]
        frame_num = len(frame_basenames)

        if ids is None:
            ids = np.random.choice(frame_num, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, frame_num, expand_ratio=8)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        image_paths = []
        depth_paths = []
        pose_paths = []

        for idx in ids:
            base = frame_basenames[idx]
            image_path = base + ".color.png"
            depth_path = base + ".depth.png"
            pose_path = base + ".pose.txt"

            image = read_image_cv2(image_path)

            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f"Depth file not found or unreadable: {depth_path}")
            depth_map = depth_raw.astype(np.float32) / self.depth_scale
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            pose_44 = np.loadtxt(pose_path).astype(np.float32)
            if pose_44.shape != (4, 4):
                raise ValueError(f"Invalid pose shape at {pose_path}: {pose_44.shape}")

            if self.pose_is_c2w:
                w2c_44 = np.linalg.inv(pose_44)
            else:
                w2c_44 = pose_44
            extri_opencv = w2c_44[:3, :]

            intri_opencv = np.eye(3, dtype=np.float32)
            intri_opencv[0, 0] = self.fx
            intri_opencv[1, 1] = self.fy
            intri_opencv[0, 2] = self.cx
            intri_opencv[1, 2] = self.cy

            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
            image_paths.append(image_path)
            depth_paths.append(depth_path)
            pose_paths.append(pose_path)

        batch = {
            "seq_name": "7scenes_" + seq_info["seq_name"],
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "image_paths": image_paths,
            "depth_paths": depth_paths,
            "pose_paths": pose_paths,
        }
        return batch
