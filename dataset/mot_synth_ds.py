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
import utils.geometric_generator_3d as gg

import utils
from conf import Conf

import platform

is_windows = any(platform.win32_ver())

# 14 useful joints for the MOTSynth dataset
from test_metrics import joint_det_metrics

USEFUL_JOINTS = [0, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21]

# number of sequences
N_SEQUENCES = 256

# number frame in each sequence
N_FRAMES_IN_SEQ = 900

# number of frames used for training in each sequence
N_SELECTED_FRAMES = 180

# maximum MOTSynth camera distance [m]
MAX_CAM_DIST = 100

Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]


class MOTSynthDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: 3D heatmap form the JTA dataset
    * y: json-list of gaussian centers (order: D, H, W)
    """

    def __init__(self, mode, cnf=None, debug=False):
        # type: (str, Conf, debug) -> None
        """
        :param mode: values in {'train', 'val'}
        :param cnf: configuration object
        """
        self.cnf = cnf
        self.mode = mode
        self.debug = debug
        assert mode in {'train', 'val', 'test'}, '`mode` must be \'train\' or \'val\''

        self.mots_ds = None
        path_to_anns = None

        if self.mode == 'train':
            if is_windows:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotations', '000.json')
            else:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotation_groups',
                                         'MOTSynth_annotations_10_train.json')
        if self.mode in ('val', 'test'):
            if is_windows:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotations', '002.json')
            else:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotation_groups',
                                         'MOTSynth_annotations_10_test.json')
        print(f'Annotation path to load: {path_to_anns}')
        self.mots_ds = MOTS(path_to_anns)

        self.catIds = self.mots_ds.getCatIds(catNms=['person'])
        self.imgIds = self.mots_ds.getImgIds(catIds=self.catIds)


        self.g = (self.cnf.sigma * 5 + 1) if (self.cnf.sigma * 5) % 2 == 0 else self.cnf.sigma * 5
        self.geometric_gen = gg.GeometricGenerator3D(self.cnf)
        self.gaussian_patch = self.geometric_gen.make_a_gaussian(
            w=self.g, h=self.g, d=self.g,
            center=(self.g // 2, self.g // 2, self.g // 2),
            s=self.cnf.sigma, device='cpu'
        )

    def __len__(self):
        # type: () -> int
        return len(self.imgIds)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, str, str, Tuple[Float, Float, Float]]

        # select sequence name and frame number
        img = self.mots_ds.loadImgs(self.imgIds[i])[0]

        # load corresponding data
        ann_ids = self.mots_ds.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.mots_ds.loadAnns(ann_ids)

        # augmentation initialization (rescale + crop)
        h, w, d = self.cnf.hmap_h, self.cnf.hmap_w, self.cnf.hmap_d
        aug_scale = np.random.uniform(0.5, 2)
        aug_h = np.random.uniform(0, 1)
        aug_w = np.random.uniform(0, 1)
        aug_offset_h = aug_h * (h * aug_scale - h)
        aug_offset_w = aug_w * (w * aug_scale - w)

        all_hmaps = []
        y = []

        for jtype in USEFUL_JOINTS:

            # empty hmap
            x = torch.zeros((self.cnf.hmap_d, self.cnf.hmap_h, self.cnf.hmap_w)).to('cpu')

            joints = MOTSynthDS.get_joints_from_anns(anns, jtype)

            if self.debug:
                self.visualize3djoint_in2d(joints, scale=True, draw_rect=(aug_scale, aug_offset_w, aug_offset_h))
            # for each joint of the same pose jtype
            for joint in joints:
                if joint['x2d'] < 0 or joint['y2d'] < 0 or joint['x2d'] > 1920 or joint['y2d'] > 1080:
                    continue

                # from GTA space to heatmap space
                cam_dist = np.sqrt(joint['x3d'] ** 2 + joint['y3d'] ** 2 + joint['z3d'] ** 2)
                cam_dist = cam_dist * ((self.cnf.hmap_d - 1) / MAX_CAM_DIST)
                joint['x2d'] /= 8
                joint['y2d'] /= 8

                # augmentation (rescale + crop)
                joint['x2d'] = joint['x2d'] * aug_scale - aug_offset_w
                joint['y2d'] = joint['y2d'] * aug_scale - aug_offset_h
                cam_dist = cam_dist / aug_scale

                center = [
                    int(round(joint['x2d'])),
                    int(round(joint['y2d'])),
                    int(round(cam_dist))
                ]

                # ignore the point if due to augmentation the point goes out of the screen
                if min(center) < 0 or joint['x2d'] > w or joint['y2d'] > h or cam_dist > d:
                    continue

                center = center[::-1]

                xa, ya, za = max(0, center[2] - self.g // 2), max(0, center[1] - self.g // 2), max(0, center[
                    0] - self.g // 2)
                xb, yb, zb = min(center[2] + self.g // 2, self.cnf.hmap_w - 1), min(center[1] + self.g // 2,
                                                                                    self.cnf.hmap_h - 1), min(
                    center[0] + self.g // 2, self.cnf.hmap_d - 1)
                hg, wg, dg = (yb - ya) + 1, (xb - xa) + 1, (zb - za) + 1

                gxa, gya, gza = 0, 0, 0
                gxb, gyb, gzb = self.g - 1, self.g - 1, self.g - 1

                if center[2] - self.g // 2 < 0:
                    gxa = -(center[2] - self.g // 2)
                if center[1] - self.g // 2 < 0:
                    gya = -(center[1] - self.g // 2)
                if center[0] - self.g // 2 < 0:
                    gza = -(center[0] - self.g // 2)
                if center[2] + self.g // 2 > (self.cnf.hmap_w - 1):
                    gxb = wg - 1
                if center[1] + self.g // 2 > (self.cnf.hmap_h - 1):
                    gyb = hg - 1
                if center[0] + self.g // 2 > (self.cnf.hmap_d - 1):
                    gzb = dg - 1

                x[za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(
                    torch.cat(tuple([
                        x[za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0),
                        self.gaussian_patch[gza:gzb + 1, gya:gyb + 1, gxa:gxb + 1].unsqueeze(0)
                    ])), 0)[0]

                y.append([USEFUL_JOINTS.index(jtype)] + center)
            if self.debug:
                self.visualize3djoint_in2d(joints)
            all_hmaps.append(x)

        y = json.dumps(y)

        return torch.cat(tuple([h.unsqueeze(0) for h in all_hmaps])), y, img['file_name'],  (aug_scale, aug_h, aug_w)

    @staticmethod
    def get_joints_from_anns(anns, jtype):
        joints = []
        for ann in anns:
            joints.append({
                'x2d': ann['keypoints'][3 * jtype],
                'y2d': ann['keypoints'][3 * jtype + 1],
                'x3d': ann['keypoints_3d'][4 * jtype],
                'y3d': ann['keypoints_3d'][4 * jtype + 1],
                'z3d': ann['keypoints_3d'][4 * jtype + 2],
                'visibility': ann['keypoints_3d'][4 * jtype + 3],
                # visibility=0: not labeled (in which case x=y=z=0),
                # visibility=1: labeled but not visible
                # visibility=2: labeled and visible.
            })
        return joints

    def get_dataset_max_cam_len(self):
        curr_max = 0
        for i in range(len(self.imgIds)):
            img = self.mots_ds.loadImgs(self.imgIds[i])[0]

            # load corresponding data
            ann_ids = self.mots_ds.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
            anns = self.mots_ds.loadAnns(ann_ids)
            for ann in anns:
                for jtype_index in range(len(ann['keypoints_3d']) // 4):
                    _max = np.sqrt(ann['keypoints_3d'][jtype_index] ** 2 + ann['keypoints_3d'][jtype_index + 1] ** 2 +
                                   ann['keypoints_3d'][jtype_index + 2] ** 2)
                    curr_max = max(_max, curr_max)
        return curr_max

    def visualize3djoint_in2d(self, joints, scale=None, draw_rect=None):
        image = np.zeros((self.cnf.hmap_h, self.cnf.hmap_w, 1), np.uint8)
        import cv2

        # draw where we are going to crop
        if draw_rect:
            aug_scale, aug_offset_w, aug_offset_h = draw_rect
            t_l = int(aug_offset_w // aug_scale), int(aug_offset_h // aug_scale)
            b_r = int((self.cnf.hmap_w + aug_offset_w) // aug_scale), int((self.cnf.hmap_h + aug_offset_h) // aug_scale)
            image = cv2.rectangle(image, t_l, b_r, 60, 2)
        for joint in joints:
            x, y = joint['x2d'], joint['y2d']
            if scale:
                x, y = x / 8, y / 8
            x, y = int(round(x)), int(round(y))
            if x < 0 or y < 0 or x >= self.cnf.hmap_w or y >= self.cnf.hmap_h:
                continue
            image[y, x, 0] = 255

        cv2.imshow('image', image)
        cv2.waitKey(0)


def main():
    import utils
    cnf = Conf(exp_name='default')
    ds = MOTSynthDS(mode='train', cnf=cnf, debug=True)
    loader = DataLoader(dataset=ds, batch_size=1, num_workers=0, shuffle=False)

    for i, sample in enumerate(loader):
        x, y, _ = sample
        x = x.to(cnf.device)
        y = json.loads(y[0])

        utils.visualize_3d_hmap(x[0, 13])
        y_pred = utils.get_multi_local_maxima_3d(hmaps3d=x.squeeze(), threshold=0.1, device=cnf.device)
        metrics = joint_det_metrics(points_pred=y_pred, points_true=y, th=1)
        f1 = metrics['f1']
        print(f'f1 score = {f1}')
        print(f'({i}) Dataset example: x.shape={tuple(x.shape)}, y={y}')


if __name__ == '__main__':
    main()
