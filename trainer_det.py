# -*- coding: utf-8 -*-
# ---------------------

import importlib
import json
import math
from datetime import datetime
from time import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import imgaug.augmenters as iaa

import matplotlib.pyplot as plt

from utils import utils
from conf import Conf
from dataset.mot_synth_ds import MOTSynthDS
from dataset.mot_synth_det_ds import MOTSynthDetDS
from test_metrics import joint_det_metrics
from models.vha_det_simple import Autoencoder as AutoencoderSimple
# from models.vha_det_complete import Autoencoder as AutoencoderPretrained
from utils.trainer_base import TrainerBase

MEAN_WIDTH = 52.61727208688375
MEAN_HEIGHT = 116.20363004472165
STD_DEV_WIDTH = 102.74863634249581
STD_DEV_HEIGHT = 135.07407255355838


class TrainerDet(TrainerBase):

    def __init__(self, cnf):
        # type: (Conf) -> TrainerJoint

        super().__init__(cnf)

        # init model
        self.model = AutoencoderSimple(hmap_d=cnf.hmap_d).to(cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)

        # init train loader
        training_set = MOTSynthDetDS(mode='train', cnf=cnf)
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size, num_workers=cnf.n_workers, shuffle=True
        )

        # init val loader
        val_set = MOTSynthDetDS(mode='val', cnf=cnf)
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=1, num_workers=cnf.n_workers, shuffle=False
        )

        # init logging stuff
        self.log_path = cnf.exp_log_path
        tb_logdir = cnf.project_log_path.abspath()
        print(f'tensorboard --logdir={tb_logdir}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []

        # starting values values
        self.epoch = 0
        self.best_val_f1_center = None

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=torch.device('cpu'))
            print('[loading checkpoint \'{}\']'.format(ck_path))
            self.epoch = ck['epoch']
            self.model.load_state_dict(ck['model'], strict=False)
            self.model.to(self.cnf.device)
            self.best_val_f1_center = ck['best_val_f1']
            if ck.get('optimizer', None) is not None:
                self.optimizer.load_state_dict(ck['optimizer'])
        else:
            dict = torch.load('log/pretrained/best.pth', map_location=torch.device('cpu'))
            self.model.load_state_dict(dict, strict=False)

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1_center
        }
        torch.save(ck, self.log_path / 'training.ck')

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        self.model.train()
        self.model.requires_grad(True)

        start_time = time()
        times = []
        t = time()
        for step, sample in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            x = sample.to(self.cnf.device)

            y_pred = self.model.forward(x)

            loss = nn.MSELoss()(y_pred, x)
            loss.backward()
            self.train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            progress = (step + 1) / self.cnf.epoch_len
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
            times.append(time() - t)
            t = time()
            if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
                print('\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s'.format(
                    datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
                    progress_bar, 100 * progress,
                    np.mean(self.train_losses), 1 / np.mean(times),
                    e=math.ceil(math.log10(self.cnf.epochs)),
                    s=math.ceil(math.log10(self.log_freq)),
                ), end='')

            if step >= self.cnf.epoch_len - 1:
                break

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')

    def test(self):
        """
        test model on the Test-Set
        """

        self.model.eval()
        self.model.requires_grad(False)

        val_f1s = {'f1_center': [], 'f1_width': [], 'f1_height': []}
        val_losses = []

        t = time()
        for step, sample in enumerate(self.val_loader):
            hmap_true, y_true, file_name, aug_info = sample
            hmap_true = hmap_true.to(self.cnf.device)
            y_true = json.loads(y_true[0])

            hmap_pred = self.model.forward(hmap_true)

            loss = nn.MSELoss()(hmap_pred, hmap_true)
            val_losses.append(loss.item())

            x_center = hmap_pred[0, 0]
            x_width = hmap_pred[0, 1]
            x_height = hmap_pred[0, 2]

            y_center = [(coord[0], coord[1], coord[2]) for coord in y_true]
            y_width = [(coord[0], coord[1], coord[2], coord[3]) for coord in y_true]
            y_height = [(coord[0], coord[1], coord[2], coord[4]) for coord in y_true]

            y_center_pred = utils.local_maxima_3d(heatmap=x_center, threshold=0.1, device=self.cnf.device)
            y_width_pred = []
            y_height_pred = []
            for center_coord in y_center_pred:
                cam_dist, y2d, x2d = center_coord

                width = float(x_width[cam_dist, y2d, x2d])
                height = float(x_height[cam_dist, y2d, x2d])

                # denormalize width and height
                width = int(round(width * STD_DEV_WIDTH + MEAN_WIDTH))
                height = int(round(height * STD_DEV_HEIGHT + MEAN_HEIGHT))

                y_width_pred.append((*center_coord, width))
                y_height_pred.append((*center_coord, height))

            metrics_center = joint_det_metrics(points_pred=y_center_pred, points_true=y_center, th=1)
            metrics_width = joint_det_metrics(points_pred=y_width_pred, points_true=y_width, th=1)
            metrics_height = joint_det_metrics(points_pred=y_height_pred, points_true=y_height, th=1)
            f1_center = metrics_center['f1']
            f1_width = metrics_width['f1']
            f1_height = metrics_height['f1']
            val_f1s['f1_center'].append(f1_center)
            val_f1s['f1_width'].append(f1_width)
            val_f1s['f1_height'].append(f1_height)

            if step >= self.cnf.test_len:
                break

        # log average f1 on test set
        mean_val_loss = np.mean(val_losses)
        mean_val_f1_center = np.mean(val_f1s['f1_center'])
        mean_val_f1_width = np.mean(val_f1s['f1_width'])
        mean_val_f1_height = np.mean(val_f1s['f1_height'])
        print(f'[TEST] AVG-Loss: {mean_val_loss:.6f}, '
              f'AVG-F1_center: {mean_val_f1_center:.6f}, '
              f'AVG-F1_width: {mean_val_f1_width:.6f}, '
              f'AVG-F1_height: {mean_val_f1_height:.6f}'
              f' │ Test time: {time() - t:.2f} s')
        self.sw.add_scalar(tag='val_F1', scalar_value=mean_val_f1_center, global_step=self.epoch)
        self.sw.add_scalar(tag='val_F1_width', scalar_value=mean_val_f1_width, global_step=self.epoch)
        self.sw.add_scalar(tag='val_F1_height', scalar_value=mean_val_f1_height, global_step=self.epoch)
        self.sw.add_scalar(tag='val_loss', scalar_value=mean_val_loss, global_step=self.epoch)

        # save best model
        if self.best_val_f1_center is None or mean_val_f1_center < self.best_val_f1_center:
            self.best_val_f1_center = mean_val_f1_center
            torch.save(self.model.state_dict(), self.log_path / 'best.pth')
