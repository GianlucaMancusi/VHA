# -*- coding: utf-8 -*-
# ---------------------

import json
import random
from typing import *

import numpy as np
import torch
from path import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO as MOTS
from os import path

import utils
from conf import Conf

# 14 useful joints for the MOTSynth dataset
USEFUL_JOINTS = [0, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21]

# number of sequences
N_SEQUENCES = 256

# number frame in each sequence
N_FRAMES_IN_SEQ = 900

# number of frames used for training in each sequence
N_SELECTED_FRAMES = 180

# maximum MOTSynth camera distance [m]
MAX_CAM_DIST = 9999

Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]


class MOTSynthDS(Dataset):
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
        :param sigma: parameter that controls the "spread" of the 3D gaussians:param cnf: configuration object
        """
        self.cnf = cnf
        self.mode = mode
        assert mode in {'train', 'val', 'test'}, '`mode` must be \'train\' or \'val\''

        self.mots_ds = MOTS(path.join(self.cnf.mot_synth_path, 'annotation_groups', 'group_0_60.json'))
        self.catIds = self.mots_ds.getCatIds(catNms=['person'])
        self.imgIds = self.mots_ds.getImgIds(catIds=self.catIds)

        self.g = (self.cnf.sigma * 5 + 1) if (self.cnf.sigma * 5) % 2 == 0 else self.cnf.sigma * 5
        self.gaussian_patch = utils.gkern(
            w=self.g, h=self.g, d=self.g,
            center=(self.g // 2, self.g // 2, self.g // 2),
            s=self.cnf.sigma, device='cpu'
        )

    def __len__(self):
        # type: () -> int
        return len(self.imgIds)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, str, str]

        # select sequence name and frame number
        img = self.mots_ds.loadImgs(self.imgIds[i])[0]
        image_H, image_W = img['height'], img['width']

        # load corresponding data
        annIds = self.mots_ds.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.mots_ds.loadAnns(annIds)

        all_hmaps = []
        y = []

        for jtype in USEFUL_JOINTS:

            # empty hmap
            x = torch.zeros((self.cnf.hmap_d, self.cnf.hmap_h, self.cnf.hmap_w)).to('cpu')

            joints = MOTSynthDS.get_joints_from_anns(anns, jtype)

            # for each joint of the same pose jtype
            for joint in joints:
                if joint['x2d'] < 0 or joint['y2d'] < 0 or joint['x2d'] > 1920 or joint['y2d'] > 1080:
                    continue

                cam_dist = np.sqrt(joint['x3d'] ** 2 + joint['y3d'] ** 2 + joint['z3d'] ** 2)
                cam_dist = cam_dist * ((self.cnf.hmap_d - 1) / MAX_CAM_DIST)

                center = [
                    int(round(joint['x2d'] / 8)),
                    int(round(joint['y2d'] / 8)),
                    int(round(cam_dist))
                ]

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

            all_hmaps.append(x)

        y = json.dumps(y)

        return torch.cat(tuple([h.unsqueeze(0) for h in all_hmaps])), y, 'NOT_AVAILABLE'

    @staticmethod
    def get_joints_from_anns(anns, jtype):
        joints = []
        for ann in anns:
            joints.append({
                'x2d': ann['keypoints'][2 * jtype],
                'y2d': ann['keypoints'][2 * jtype + 1],
                'x3d': ann['keypoints_3d'][3 * jtype],
                'y3d': ann['keypoints_3d'][3 * jtype + 1],
                'z3d': ann['keypoints_3d'][3 * jtype + 2],
                'visibility': ann['keypoints_3d'][3 * jtype + 3],
                # visibility=0: not labeled (in which case x=y=z=0), visibility=1: labeled but not visible, and visibility=2: labeled and visible.
            })
        return joints


def main():
    import utils
    cnf = Conf(exp_name='default')
    ds = MOTSynthDS(mode='val', cnf=cnf)
    loader = DataLoader(dataset=ds, batch_size=1, num_workers=0, shuffle=False)

    for i, sample in enumerate(loader):
        x, y, _ = sample
        y = json.loads(y[0])

        utils.visualize_3d_hmap(x[0, 0])
        print(f'({i}) Dataset example: x.shape={tuple(x.shape)}, y={y}')


if __name__ == '__main__':
    main()