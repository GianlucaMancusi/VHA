# -*- coding: utf-8 -*-
# ---------------------

import json
from os import path
from typing import *

import numpy as np
import torch
from pycocotools.coco import COCO as MOTS
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa

import utils.geometric_generator_3d as gg
from conf import Conf

from utils import utils

import platform

# maximum MOTSynth camera distance [m]
MAX_CAM_DIST = 100

# camera intrinsic parameters: fx, fy, cx, cy
CAMERA_PARAMS = 1158, 1158, 960, 540

MEAN_WIDTH = 52.61727208688375
MEAN_HEIGHT = 116.20363004472165
STD_DEV_WIDTH = 102.74863634249581
STD_DEV_HEIGHT = 135.07407255355838


class MOTSynthDetDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: 3D heatmap form the JTA dataset
    * y: json-list of gaussian centers (order: D, H, W)
    """

    def __init__(self, mode, cnf=None):
        # type: (str, Conf) -> None
        """
        :param mode: values in {'train', 'val'}
        :param cnf: configuration object
        """
        self.mode = mode
        self.cnf = cnf
        assert self.mode in {'train', 'val', 'test'}, '`mode` must be \'train\' or \'val\''

        is_windows = any(platform.win32_ver())

        self.mots_ds = None
        path_to_anns = None

        if self.mode == 'train':
            if is_windows:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotations', '123.json')
                # path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotation_groups',
                #                         'MOTSynth_annotations_10_test.json')
            else:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotation_groups',
                                         'MOTSynth_annotations_10_train.json')
        if self.mode in ('val', 'test'):
            if is_windows:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotations', '275.json')
            else:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotation_groups',
                                         'MOTSynth_annotations_10_test.json')
        print(f'Annotation path to load: {path_to_anns}')
        self.mots_ds = MOTS(path_to_anns)

        self.catIds = self.mots_ds.getCatIds(catNms=['person'])
        self.imgIds = self.mots_ds.getImgIds(catIds=self.catIds)

        # max_cam_dist = self.get_dataset_max_cam_len()

        self.g = (self.cnf.sigma * 5 + 1) if (self.cnf.sigma * 5) % 2 == 0 else self.cnf.sigma * 5
        self.geometric_gen = gg.GeometricGenerator3D(self.cnf)
        self.gaussian_patch = self.geometric_gen.make_a_gaussian(
            w=self.g, h=self.g, d=self.g,
            center=(self.g // 2, self.g // 2, self.g // 2),
            s=self.cnf.sigma, device='cpu'
        )
        self.sphere_patch = self.geometric_gen.make_a_sphere(
            w=self.cnf.sphere_diameter, h=self.cnf.sphere_diameter, d=self.cnf.sphere_diameter,
            center=(self.cnf.sphere_diameter // 2, self.cnf.sphere_diameter // 2, self.cnf.sphere_diameter // 2),
            r=self.cnf.sphere_diameter // 2, device='cpu'
        )

    def __len__(self):
        # type: () -> int
        if self.mode == 'train':
            return len(self.imgIds)
        elif self.mode in ('val', 'test'):
            return self.cnf.test_len

    def __getitem__(self, i):
        # select sequence name and frame number
        img = self.mots_ds.loadImgs(self.imgIds[i])[0]

        # load corresponding data
        ann_ids = self.mots_ds.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.mots_ds.loadAnns(ann_ids)

        augmentation = self.cnf.data_augmentation if self.mode == 'train' else 'no'
        x_tensor, aug_info, y_coords3d, y_coords2d = self.generate_3d_tensors(anns, augmentation=augmentation)

        if self.mode == 'train':
            return x_tensor, img['file_name'], aug_info
        elif self.mode in ('val', 'test'):
            return x_tensor, y_coords2d, img['file_name'], aug_info

    def generate_3d_tensors(self, anns, augmentation):
        # type: (List[dict], str) -> Tuple[Tensor, Tuple[float, float, float], Any, Any]
        # augmentation initialization (rescale + crop)

        h, w, d = self.cnf.hmap_h, self.cnf.hmap_w, self.cnf.hmap_d

        aug_scale = np.random.uniform(0.5, 2) if augmentation == 'all' else 1
        aug_h = np.random.uniform(0, 1) if augmentation == 'all' else 0
        aug_w = np.random.uniform(0, 1) if augmentation == 'all' else 0
        aug_offset_h = aug_h * (h * aug_scale - h)
        aug_offset_w = aug_w * (w * aug_scale - w)

        y_coords3d = []
        y_coords2d = []

        # empty hmap
        x_centers = torch.zeros((self.cnf.hmap_d, self.cnf.hmap_h, self.cnf.hmap_w)).to('cpu')
        x_width = torch.zeros((self.cnf.hmap_d, self.cnf.hmap_h, self.cnf.hmap_w)).to('cpu')
        x_height = torch.zeros((self.cnf.hmap_d, self.cnf.hmap_h, self.cnf.hmap_w)).to('cpu')

        bboxes = MOTSynthDetDS.get_bboxes_from_anns(anns)

        # for each joint of the same pose jtype
        for bbox in bboxes:
            person_center_real = {'x2d': bbox['x2d'] + bbox['width'] // 2, 'y2d': bbox['y2d'] + bbox['height'] // 2}

            # from GTA space to heatmap space
            cam_dist_real = np.sqrt(bbox['mean_x3d'] ** 2 + bbox['mean_y3d'] ** 2 + bbox['mean_z3d'] ** 2)
            cam_dist_in_map = cam_dist_real * ((self.cnf.hmap_d - 1) / MAX_CAM_DIST)
            person_center_in_map = {'x2d': person_center_real['x2d'] / 8, 'y2d': person_center_real['y2d'] / 8}

            # augmentation (rescale + crop)
            if augmentation == 'all':
                person_center_in_map['x2d'] = person_center_in_map['x2d'] * aug_scale - aug_offset_w
                person_center_in_map['y2d'] = person_center_in_map['y2d'] * aug_scale - aug_offset_h
                cam_dist_in_map /= aug_scale

            center = [
                int(round(person_center_in_map['x2d'])),
                int(round(person_center_in_map['y2d'])),
                int(round(cam_dist_in_map))
            ]

            # ignore the point if due to augmentation the point goes out of the screen
            if min(center) < 0 or person_center_in_map['x2d'] > w or person_center_in_map['y2d'] > h \
                or cam_dist_in_map > d:
                continue

            width = bbox['width'] * aug_scale
            height = bbox['height'] * aug_scale
            normalized_width = (width - MEAN_WIDTH) / STD_DEV_WIDTH
            normalized_height = (height - MEAN_HEIGHT) / STD_DEV_HEIGHT

            # Update the center, width and height tensor
            self.geometric_gen.paste_tensor(x_centers, self.gaussian_patch, self.g, center)  # in place function
            self.geometric_gen.paste_tensor(x_width, self.sphere_patch, self.cnf.sphere_diameter,
                                            center=center, mul_value=normalized_width, use_max=False)
            self.geometric_gen.paste_tensor(x_height, self.sphere_patch, self.cnf.sphere_diameter,
                                            center=center, mul_value=normalized_height, use_max=False)

            y_coords3d.append([bbox['mean_x3d'], bbox['mean_y3d'], bbox['mean_z3d']])
            y_coords2d.append([int(round(cam_dist_in_map)),
                               int(round(person_center_in_map['y2d'])),
                               int(round(person_center_in_map['x2d'])),
                               width, height
                               ])
        y_coords3d = json.dumps(y_coords3d)
        y_coords2d = json.dumps(y_coords2d)

        # Merge the 3 tensors in one only
        x_tensor = torch.cat(tuple([
            x_centers.unsqueeze(0),
            x_width.unsqueeze(0),
            x_height.unsqueeze(0)
        ]))

        return x_tensor, (aug_scale, aug_h, aug_w), y_coords3d, y_coords2d

    def get_frame(self, file_path):
        # read input frame
        frame_path = self.cnf.mot_synth_path / file_path
        frame = utils.imread(frame_path).convert('RGB')
        # frame = transforms.ToTensor()(frame)
        return frame

    @staticmethod
    def get_bboxes_from_anns(anns):
        from statistics import mean

        def visibility_condition(keypoint):
            return keypoint[2] == 2 and keypoint[0] < 1920 and keypoint[1] < 1080 and keypoint[0] > 0 and keypoint[
                1] > 0

        bboxes = []
        for ann in anns:
            keypoints_list = [ann['keypoints'][n:n + 3] for n in range(0, len(ann['keypoints']), 3)]
            n_visible_joints = np.array([visibility_condition(keypoint) for keypoint in keypoints_list]).sum()
            if n_visible_joints > ann['num_keypoints'] * 0.25:
                bboxes.append({
                    'x2d': ann['bbox'][0],
                    'y2d': ann['bbox'][1],
                    'width': ann['bbox'][2],
                    'height': ann['bbox'][3],
                    'mean_x3d': mean([ann['keypoints_3d'][4 * jtype] for jtype in range(ann['num_keypoints'])]),
                    'mean_y3d': mean([ann['keypoints_3d'][4 * jtype + 1] for jtype in range(ann['num_keypoints'])]),
                    'mean_z3d': mean([ann['keypoints_3d'][4 * jtype + 2] for jtype in range(ann['num_keypoints'])]),
                })
        return bboxes


def main():
    from test_metrics import joint_det_metrics, compute_det_metrics_iou
    cnf = Conf(exp_name='vha_d_c3d_debug')

    # load dataset
    mode = 'test'
    ds = MOTSynthDetDS(mode=mode, cnf=cnf)
    loader = DataLoader(dataset=ds, batch_size=1, num_workers=0, shuffle=False)

    # load model
    from models.vha_det_c3d_pretrained import Autoencoder as AutoencoderC3dPretrained
    model = AutoencoderC3dPretrained(hmap_d=cnf.hmap_d, legacy_pretrained=cnf.saved_epoch == 0).to(cnf.device)
    model.eval()
    model.requires_grad(False)
    if cnf.model_weights is not None:
        model.load_state_dict(cnf.model_weights, strict=False)

    # ======== MAIN LOOP ========
    for i, sample in enumerate(loader):
        x, y, file_name, aug_info = None, None, None, None

        if mode == 'test':
            x, y, file_name, aug_info = sample
        if mode == 'train':
            x, file_name, aug_info = sample
        x = x.to(cnf.device)
        x_center, x_width, x_height = x[0, 0], x[0, 1], x[0, 2]

        y_pred = model.forward(x)
        x_pred_center, x_pred_width, x_pred_height = y_pred[0, 0], y_pred[0, 1], y_pred[0, 2]

        if mode == 'test':
            y = json.loads(y[0])
            y_center = [(coord[0], coord[1], coord[2]) for coord in y]
            y_width = [(coord[0], coord[1], coord[2], coord[3]) for coord in y]
            y_height = [(coord[0], coord[1], coord[2], coord[4]) for coord in y]

        # utils.visualize_3d_hmap(x[0, 2])
        y_center_pred = utils.local_maxima_3d(heatmap=x_center, threshold=0.1, device=cnf.device)
        y_width_pred = []
        y_height_pred = []
        bboxes_info_pred = []
        for cam_dist, y2d, x2d in y_center_pred:
            width = float(x_width[cam_dist, y2d, x2d])
            height = float(x_height[cam_dist, y2d, x2d])

            # denormalize width and height
            width = int(round(width * STD_DEV_WIDTH + MEAN_WIDTH))
            height = int(round(height * STD_DEV_HEIGHT + MEAN_HEIGHT))

            y_width_pred.append((cam_dist, y2d, x2d, width))
            y_height_pred.append((cam_dist, y2d, x2d, height))

            x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=cnf.q)
            bboxes_info_pred.append((x2d - width / 2, y2d - height / 2, width, height, cam_dist))

        if mode == 'test':
            bboxes_info_true = []
            for cam_dist, y2d, x2d, width, height in y:
                x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=cnf.q)
                bboxes_info_true.append((x2d - width / 2, y2d - height / 2, width, height, cam_dist))

            metrics_iou = compute_det_metrics_iou(bboxes_info_pred, bboxes_info_true)
            metrics_center = joint_det_metrics(points_pred=y_center_pred, points_true=y_center, th=1)
            metrics_width = joint_det_metrics(points_pred=y_width_pred, points_true=y_width, th=1)
            metrics_height = joint_det_metrics(points_pred=y_height_pred, points_true=y_height, th=1)
            f1_iou = metrics_iou['f1']
            f1_center = metrics_center['f1']
            f1_width = metrics_width['f1']
            f1_height = metrics_height['f1']
            print(f'f1_center={f1_iou}, f1_center={f1_center}, f1_width={f1_width}, f1_height={f1_height}')
        # print(f'({i}) Dataset example: x.shape={tuple(x.shape)}, y={y}')

        img_original = np.array(utils.imread(cnf.mot_synth_path / file_name[0]).convert("RGB"))
        # out_path = cnf.exp_log_path / f'DS_DEBUG_{i}_bboxes_pred.jpg'
        utils.visualize_bboxes(img_original, bboxes_info_pred, use_z=True, half_images=True, aug_info=aug_info)


if __name__ == '__main__':
    main()
