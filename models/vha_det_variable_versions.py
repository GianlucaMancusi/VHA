# -*- coding: utf-8 -*-
# ---------------------
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from models import BaseModel


class Upsample(nn.Module):
    """
    Upsamples a given tensor by (scale_factor)X.
    """

    def __init__(self, scale_factor=2, mode='trilinear'):
        # type: (int, str) -> Upsample
        """
        :param scale_factor: the multiplier for the image height / width
        :param mode: the upsampling algorithm - values in {'nearest', 'linear', 'bilinear', 'trilinear'}
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)

    def extra_repr(self):
        return f'scale_factor={self.scale_factor}, mode={self.mode}'


# ---------------------

class Autoencoder(BaseModel):
    """
    VHA: (V)olumetric (H)eatmap (A)utoencoder
    VHAv1: (d1, d2, d3) = (1, 2, 2) and (s1, s2, s3) = (1, 2, 1);
    VHAv2: (d1, d2, d3) = (2, 4, 4) and (s1, s2, s3) = (2, 2, 1);
    VHAv3: (d1, d2, d3) = (2, 4, 8) and (s1, s2, s3) = (2, 2, 2);
    """

    def __init__(self, vha_version, hmap_d=316):
        # type: (int) -> None
        """
        VHAv1: (d1, d2, d3) = (1, 2, 2) and (s1, s2, s3) = (1, 2, 1);
        VHAv2: (d1, d2, d3) = (2, 4, 4) and (s1, s2, s3) = (2, 2, 1);
        VHAv3: (d1, d2, d3) = (2, 4, 8) and (s1, s2, s3) = (2, 2, 2);
        :param vha_version: Available versions = 1, 2, 3
        :param hmap_d: number of input channels
        """

        super().__init__()
        d1, d2, d3, s1, s2, s3 = 0, 0, 0, 0, 0, 0
        """
        VHAv1: (d1, d2, d3) = (1, 2, 2) and (s1, s2, s3) = (1, 2, 1); 
        VHAv2: (d1, d2, d3) = (2, 4, 4) and (s1, s2, s3) = (2, 2, 1);
        VHAv3: (d1, d2, d3) = (2, 4, 8) and (s1, s2, s3) = (2, 2, 2);
        """
        assert 1 <= vha_version <= 3, "Available vha_version are 1, 2, 3."
        if vha_version == 1:
            d1, d2, d3 = (1, 2, 2)
            s1, s2, s3 = (1, 2, 1)
        elif vha_version == 2:
            d1, d2, d3 = (2, 4, 4)
            s1, s2, s3 = (2, 2, 1)
        elif vha_version == 3:
            d1, d2, d3 = (2, 4, 8)
            s1, s2, s3 = (2, 2, 2)


        self.fuser_c3d = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d, out_channels=hmap_d // d1, kernel_size=5, stride=s1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d1, out_channels=hmap_d // d2, kernel_size=5, stride=s2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d2, out_channels=hmap_d // d3, kernel_size=5, stride=s3, padding=2),
            nn.ReLU(True),
        )
        self.encoder_w_h = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d, out_channels=hmap_d // d1, kernel_size=5, stride=s1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d1, out_channels=hmap_d // d2, kernel_size=5, stride=s2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d2, out_channels=hmap_d // d3, kernel_size=5, stride=s3, padding=2),
            nn.ReLU(True),
        )

        # --------------

        self.defuser_c3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=3, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d // d3, out_channels=hmap_d // d2, kernel_size=5, padding=2, stride=s3),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d2, out_channels=hmap_d // d1, kernel_size=5, padding=2, stride=s2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d1, out_channels=hmap_d, kernel_size=5, padding=2, stride=s1),
            nn.ReLU(True)
        )
        self.decoder_w_h = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d // d3, out_channels=hmap_d // d2, kernel_size=5, padding=2, stride=s3),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d2, out_channels=hmap_d // d1, kernel_size=5, padding=2, stride=s2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // d1, out_channels=hmap_d, kernel_size=5, padding=2, stride=s1),
            nn.ReLU(True)
        )


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x.shape = batch, 3, h, w, c
        x_c = x[:, 0, :, :, :].contiguous()
        x_c = self.encoder(x_c)

        x_wh = x[:, 1:3, :, :, :]
        x_wh = torch.reshape(x_wh, (x_wh.shape[0] * 2, x_wh.shape[2], x_wh.shape[3], x_wh.shape[4])).contiguous()
        x_wh = self.encoder_w_h(x_wh)

        # batch * 3, h, w, c
        x = torch.cat(tuple([x_c, x_wh]), dim=0)
        # batch, 3, h, w, c
        x = torch.reshape(x, (x.shape[0] // 3, 3, x.shape[1], x.shape[2], x.shape[3])).contiguous()

        x = self.fuser_c3d(x)
        return x

    def decode(self, x):
        x = self.defuser_c3d(x)

        # x.shape = batch, 3, h, w, c
        x_c = x[:, 0, :, :, :].contiguous()
        x_c = self.decoder(x_c)

        x_wh = x[:, 1:3, :, :, :]
        x_wh = torch.reshape(x_wh, (x_wh.shape[0] * 2, x_wh.shape[2], x_wh.shape[3], x_wh.shape[4])).contiguous()
        x_wh = self.decoder_w_h(x_wh)

        x = torch.cat(tuple([x_c, x_wh]), dim=0)
        x = torch.reshape(x, (x.shape[0] // 3, 3, x.shape[1], x.shape[2], x.shape[3])).contiguous()
        return x

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.encode(x)
        x = self.decode(x)
        return x


# ---------------------


def main():
    from time import time
    from statistics import mean, stdev

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 3

    model = Autoencoder(vha_version=1).to(device)
    model.train()
    model.requires_grad(True)
    print(model)

    print(f'* number of parameters: {model.n_param}')

    t_list = []

    for i in range(200):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        print('\n--- ENCODER ---')
        x = torch.rand((batch_size, 3, 316, 1080 // 8, 1920 // 8)).to(device)
        start.record()
        y = model.encode(x)
        print(f'* input shape: {tuple(x.shape)}')
        print(f'* output shape: {tuple(y.shape)}')

        print('\n--- DECODER ---')
        xd = model.decode(y)
        end.record()
        torch.cuda.synchronize()
        t_list.append(start.elapsed_time(end))
        print(f'* input shape: {tuple(y.shape)}')
        print(f'* output shape: {tuple(xd.shape)}')

    print('\n--------- PROFILER ---------')
    print(f'VHA time: {mean(t_list)}ms, stdev: +-{stdev(t_list)}ms')


if __name__ == '__main__':
    main()
