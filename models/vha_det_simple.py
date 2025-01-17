# -*- coding: utf-8 -*-
# ---------------------

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
    """

    def __init__(self, hmap_d=316, legacy_pretrained=True):
        # type: (int) -> None
        """
        :param hmap_d: number of input channels
        """

        super().__init__()

        self.fuser2 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d, out_channels=hmap_d // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 2, out_channels=hmap_d // 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
        )

        # --------------

        self.defuser2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=3, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 4, kernel_size=5, padding=2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 4, out_channels=hmap_d // 2, kernel_size=5, padding=2),
            Upsample(mode='bilinear'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hmap_d // 2, out_channels=hmap_d, kernel_size=5, padding=2),
            nn.ReLU(True)
        )

        if legacy_pretrained:
            self.load_legacy_pretrained_weights()

    def load_legacy_pretrained_weights(self):
        self.load_w('log/pretrained/best.pth', strict=False, map_location=torch.device('cpu'))

    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor

        # batch * 14, h, w, c
        x = self.encoder(torch.reshape(x, (x.shape[0] * 3, x.shape[2], x.shape[3], x.shape[4])).contiguous())
        # batch, 14, h, w, c
        x = torch.reshape(x, (x.shape[0] // 3, 3, x.shape[1], x.shape[2], x.shape[3])).contiguous()

        x = self.fuser2(x)
        return x

    def decode(self, x):
        x = self.defuser2(x)

        x = self.decoder(torch.reshape(x, (x.shape[0] * 3, x.shape[2], x.shape[3], x.shape[4])).contiguous())
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

    model = Autoencoder().to(device)
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
