# -*- coding: utf-8 -*-
# ---------------------

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

from conf import Conf
from dataset.mot_synth_det_ds import MOTSynthDetDS
from models.vha_det_c3d_pretrained import Autoencoder as AutoencoderC3dPretrained
from models.vha_det_c3d_variable_code_size import Autoencoder as AutoencoderC3dVariableCode
from models.vha_det_divided import Autoencoder as AutoencoderDivided
from models.vha_det_simple import Autoencoder as AutoencoderSimple
from models.vha_det_variable_versions import Autoencoder as AutoencoderVariableVersions
from models.vha_matteo import Autoencoder
from test_metrics import joint_det_metrics, compute_det_metrics_iou
from utils import utils
from utils.MaskedMSELoss import MaskedMSELoss
from utils.trainer_base import TrainerBase

MEAN_WIDTH = 65.7656005077149 # 52.61727208688375
MEAN_HEIGHT = 145.25142926267972 # 116.20363004472165
STD_DEV_WIDTH = 137.9245315121436 # 102.74863634249581
STD_DEV_HEIGHT = 185.65746476128137 # 135.07407255355838
MAX_WIDTH = 1919
MAX_HEIGHT = 1079


def masked_mse_loss(y_pred, y_true, binary_mask):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Masked mean squared error loss.
    """
    squared_errors = (y_true - y_pred) ** 2
    masked_squared_errors = squared_errors * binary_mask
    masked_mse = masked_squared_errors.sum() / (binary_mask.sum() + 1e-10)
    return masked_mse


class TrainerDet(TrainerBase):

    def __init__(self, cnf):
        super().__init__(cnf)

        # init model
        pretrained_condition = cnf.pretrained and cnf.saved_epoch == 0

        self.model = Autoencoder(hmap_d=cnf.hmap_d, legacy_pretrained=False)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)

        # init train data_loader
        training_set = MOTSynthDetDS(mode='train', cnf=cnf)
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=cnf.batch_size, num_workers=cnf.n_workers,
            worker_init_fn=MOTSynthDetDS.wif, shuffle=True
        )

        # init validation data_loader
        val_set = MOTSynthDetDS(mode='val', cnf=cnf)
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=1, num_workers=1, shuffle=False,
            worker_init_fn=MOTSynthDetDS.wif_test
        )

        # init logging stuff
        self.log_path = cnf.exp_log_path
        tb_logdir = cnf.project_log_path.abspath()
        print(f'tensorboard --logdir={tb_logdir}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values values
        self.current_epoch = 0
        self.best_val_f1 = None

        # possibly load checkpoint
        self.load_ck()
        # self.load_previous()

        self.model.to(self.cnf.device)

    def load_ck(self):
        """
        load training checkpoint
        """
        self.current_epoch = self.cnf.saved_epoch
        if self.cnf.model_weights is not None:
            self.model.load_state_dict(self.cnf.model_weights, strict=False)

        self.best_val_f1 = self.cnf.best_val_f1
        if self.cnf.optimizer_data is not None:
            self.optimizer.load_state_dict(self.cnf.optimizer_data)
    
    def load_previous(self):
        ck_path = '/home/matteo/PycharmProjects/mancu/VHA/log/vha_s4/training.ck'
        ck = torch.load(ck_path, map_location=torch.device('cpu'))
        print('[loading checkpoint \'{}\']'.format(ck_path))
        model_weights = ck['model']
        self.model.load_state_dict(model_weights)

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1
        }
        torch.save(ck, self.log_path / 'training.ck')

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        self.model.train()
        self.model.requires_grad(True)

        start_time, t = time(), time()
        times = []
        train_losses = {'all': [], 'center': [], 'width': [], 'height': []}
        for step, sample in enumerate(self.train_loader):
            x_true, _, file_name, aug_info = sample

            self.optimizer.zero_grad()
            x_true = x_true.to(self.cnf.device)
            y_pred = self.model.forward(x_true)

            x_true_center, x_true_width, x_true_height = x_true[:, 0], x_true[:, 1], x_true[:, 2]
            x_pred_center, x_pred_width, x_pred_height = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

            mask = (x_true_height != 0).float()
            loss_center = self.cnf.masked_loss_c * nn.MSELoss()(x_pred_center, x_true_center)
            loss_width = masked_mse_loss(x_pred_width, x_true_width, mask)
            loss_height = masked_mse_loss(x_pred_height, x_true_height, mask)
            # loss_width = self.cnf.masked_loss_w * MaskedMSELoss()(x_pred_width, x_true_width, mask=mask)
            # loss_height = self.cnf.masked_loss_h * MaskedMSELoss()(x_pred_height, x_true_height, mask=mask)
            loss = loss_center + loss_width + loss_height
            loss.backward()
            self.optimizer.step(None)
            train_losses['all'].append(loss.item())
            train_losses['center'].append(loss_center.item())
            train_losses['width'].append(loss_width.item())
            train_losses['height'].append(loss_height.item())

            # print an incredible progress bar
            progress = (step + 1) / self.cnf.epoch_len
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
            times.append(time() - t)
            t = time()
            if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
                print('\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ '
                      'Loss: {:.6f} │ LossC: {:.6f} │ LossW: {:.6f} │ LossH: {:.6f} │ ↯: {:5.2f} step/s'.format(
                    datetime.now().strftime("%m-%d@%H:%M"), self.current_epoch, step + 1,
                    progress_bar, 100 * progress,
                    np.mean(train_losses['all']),
                    np.mean(train_losses['center']),
                    np.mean(train_losses['width']),
                    np.mean(train_losses['height']),
                                                                                1 / np.mean(times),
                    e=math.ceil(math.log10(self.cnf.epochs)),
                    s=math.ceil(math.log10(self.cnf.epoch_len)),
                ), end='')

            if step >= self.cnf.epoch_len - 1:
                break

        # log average loss of this epoch
        mean_epoch_loss = np.mean(train_losses['all'])
        mean_epoch_loss_center = np.mean(train_losses['center'])
        mean_epoch_loss_width = np.mean(train_losses['width'])
        mean_epoch_loss_height = np.mean(train_losses['height'])
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.current_epoch)
        self.sw.add_scalar(tag='train_loss/center', scalar_value=mean_epoch_loss_center, global_step=self.current_epoch)
        self.sw.add_scalar(tag='train_loss/width', scalar_value=mean_epoch_loss_width, global_step=self.current_epoch)
        self.sw.add_scalar(tag='train_loss/height', scalar_value=mean_epoch_loss_height, global_step=self.current_epoch)

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')

    def test(self):
        """
        test model on the Test-Set
        """

        self.model.eval()
        self.model.requires_grad(False)

        val_f1s = {'f1_iou': [], 'f1_center': [], 'f1_width': [], 'f1_height': []}
        val_losses = {'all': [], 'center': [], 'width': [], 'height': []}

        t = time()
        for step, sample in enumerate(self.val_loader):
            hmap_true, y_true, file_name, aug_info = sample
            hmap_true = hmap_true.to(self.cnf.device)
            y_true = json.loads(y_true[0])

            hmap_pred = self.model.forward(hmap_true)

            x_true_center, x_true_width, x_true_height = hmap_true[0, 0], hmap_true[0, 1], hmap_true[0, 2]
            x_pred_center, x_pred_width, x_pred_height = hmap_pred[0, 0], hmap_pred[0, 1], hmap_pred[0, 2]

            # log center, width, height losses
            mask = torch.tensor(torch.where(x_true_height != 0, 1, 0), dtype=torch.float32)
            loss_center = self.cnf.masked_loss_c * nn.MSELoss()(x_pred_center, x_true_center)
            loss_width = self.cnf.masked_loss_w * MaskedMSELoss()(x_pred_width, x_true_width, mask=mask)
            loss_height = self.cnf.masked_loss_h * MaskedMSELoss()(x_pred_height, x_true_height, mask=mask)
            loss = loss_center + loss_width + loss_height
            val_losses['all'].append(loss.item())
            val_losses['center'].append(loss_center.item())
            val_losses['width'].append(loss_width.item())
            val_losses['height'].append(loss_height.item())

            y_center = [(coord[0], coord[1], coord[2]) for coord in y_true]
            y_width = [(coord[0], coord[1], coord[2], coord[3]) for coord in y_true]
            y_height = [(coord[0], coord[1], coord[2], coord[4]) for coord in y_true]

            y_center_pred = utils.local_maxima_3d(heatmap=x_pred_center, threshold=0.1, device=self.cnf.device)
            y_width_pred = []
            y_height_pred = []
            bboxes_info_pred = []
            for center_coord in y_center_pred:  # y_center_pred
                cam_dist, y2d, x2d = center_coord

                width = float(x_pred_width[cam_dist, y2d, x2d])
                height = float(x_pred_height[cam_dist, y2d, x2d])

                # denormalize width and height
                width = int(round(width * STD_DEV_WIDTH + MEAN_WIDTH))
                height = int(round(height * STD_DEV_HEIGHT + MEAN_HEIGHT))
                # width = int(round(width * MAX_WIDTH))
                # height = int(round(height * MAX_HEIGHT))

                y_width_pred.append((*center_coord, width))
                y_height_pred.append((*center_coord, height))

                x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=self.cnf.q)
                bboxes_info_pred.append((x2d - width / 2, y2d - height / 2, width, height, cam_dist))

            y_center_true = utils.local_maxima_3d(heatmap=x_true_center, threshold=0.1, device=self.cnf.device)
            bboxes_info_true = []
            for center_coord in y_center_true:

                cam_dist, y2d, x2d = center_coord

                width = float(x_true_width[cam_dist, y2d, x2d])
                height = float(x_true_height[cam_dist, y2d, x2d])

                # denormalize width and height
                width = int(round(width * STD_DEV_WIDTH + MEAN_WIDTH))
                height = int(round(height * STD_DEV_HEIGHT + MEAN_HEIGHT))
                # width = int(round(width * MAX_WIDTH))
                # height = int(round(height * MAX_HEIGHT))

                y_width_pred.append((*center_coord, width))
                y_height_pred.append((*center_coord, height))

                x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=self.cnf.q)
                bboxes_info_true.append((x2d - width / 2, y2d - height / 2, width, height, cam_dist))

            metrics_iou = compute_det_metrics_iou(bboxes_a=bboxes_info_pred, bboxes_b=bboxes_info_true)
            metrics_center = joint_det_metrics(points_pred=y_center_pred, points_true=y_center, th=1)
            metrics_width = joint_det_metrics(points_pred=y_width_pred, points_true=y_width, th=1)
            metrics_height = joint_det_metrics(points_pred=y_height_pred, points_true=y_height, th=1)
            f1_iou = metrics_iou['f1']
            f1_center = metrics_center['f1']
            f1_width = metrics_width['f1']
            f1_height = metrics_height['f1']
            val_f1s['f1_iou'].append(f1_iou)
            val_f1s['f1_center'].append(f1_center)
            val_f1s['f1_width'].append(f1_width)
            val_f1s['f1_height'].append(f1_height)

            if step < 3:
                img_original = np.array(utils.imread(self.cnf.mot_synth_path / file_name[0]).convert("RGB"))
                hmap_pred = hmap_pred.squeeze()
                out_path = self.cnf.exp_log_path / f'{step}_center_pred.mp4'
                utils.save_3d_hmap(hmap=hmap_pred[0, ...], path=out_path)
                out_path = self.cnf.exp_log_path / f'{step}_width_pred.mp4'
                utils.save_3d_hmap(hmap=hmap_pred[1, ...], path=out_path, shift_values=True)
                out_path = self.cnf.exp_log_path / f'{step}_height_pred.mp4'
                utils.save_3d_hmap(hmap=hmap_pred[2, ...], path=out_path, shift_values=True)
                out_path = self.cnf.exp_log_path / f'{step}_bboxes_pred.jpg'
                utils.save_bboxes(img_original, bboxes_info_pred, path=out_path, use_z=True, half_images=True)

                hmap_true = hmap_true.squeeze()
                out_path = self.cnf.exp_log_path / f'{step}_center_true.mp4'
                utils.save_3d_hmap(hmap=hmap_true[0, ...], path=out_path)
                out_path = self.cnf.exp_log_path / f'{step}_width_true.mp4'
                utils.save_3d_hmap(hmap=hmap_true[1, ...], path=out_path, shift_values=True)
                out_path = self.cnf.exp_log_path / f'{step}_height_true.mp4'
                utils.save_3d_hmap(hmap=hmap_true[2, ...], path=out_path, shift_values=True)
                out_path = self.cnf.exp_log_path / f'{step}_bboxes_true.jpg'
                utils.save_bboxes(img_original, bboxes_info_true, path=out_path, use_z=True, half_images=True)

            if step >= self.cnf.test_len - 1:
                break

        # log average f1 on test set
        mean_val_loss = np.mean(val_losses['all'])
        mean_val_f1_iou = np.mean(val_f1s['f1_iou'])
        mean_val_f1_center = np.mean(val_f1s['f1_center'])
        mean_val_f1_width = np.mean(val_f1s['f1_width'])
        mean_val_f1_height = np.mean(val_f1s['f1_height'])
        mean_val_loss_center = np.mean(val_losses['center'])
        mean_val_loss_width = np.mean(val_losses['width'])
        mean_val_loss_height = np.mean(val_losses['height'])
        print(f'[TEST] AVG-Loss: {mean_val_loss:.6f}, '
              f'AVG-F1_iou: {mean_val_f1_iou:.6f}, '
              f'AVG-F1_center: {mean_val_f1_center:.6f}, '
              f'AVG-F1_width: {mean_val_f1_width:.6f}, '
              f'AVG-F1_height: {mean_val_f1_height:.6f}'
              f' │ Test time: {time() - t:.2f} s')
        self.sw.add_scalar(tag='val_F1/iou', scalar_value=mean_val_f1_iou, global_step=self.current_epoch)
        self.sw.add_scalar(tag='val_F1/center', scalar_value=mean_val_f1_center, global_step=self.current_epoch)
        self.sw.add_scalar(tag='val_F1/width', scalar_value=mean_val_f1_width, global_step=self.current_epoch)
        self.sw.add_scalar(tag='val_F1/height', scalar_value=mean_val_f1_height, global_step=self.current_epoch)
        self.sw.add_scalar(tag='val_loss', scalar_value=mean_val_loss, global_step=self.current_epoch)
        self.sw.add_scalar(tag='val_loss/center', scalar_value=mean_val_loss_center, global_step=self.current_epoch)
        self.sw.add_scalar(tag='val_loss/width', scalar_value=mean_val_loss_width, global_step=self.current_epoch)
        self.sw.add_scalar(tag='val_loss/height', scalar_value=mean_val_loss_height, global_step=self.current_epoch)

        # save best model
        if self.best_val_f1 is None or mean_val_f1_iou < self.best_val_f1:
            self.best_val_f1 = mean_val_f1_iou
            torch.save(self.model.state_dict(), self.log_path / 'best.pth')
