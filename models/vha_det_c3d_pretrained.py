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
    """

    def __init__(self, hmap_d=316, legacy_pretrained=True):
        # type: (int) -> None
        """
        :param hmap_d: number of input channels
        """

        super().__init__()

        self.fuser_c3d = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=1, kernel_size=5, padding=2),
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
        self.encoder_w_h = deepcopy(self.encoder)

        # --------------

        self.defuser_c3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=3, kernel_size=5, padding=2),
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
        self.decoder_w_h = deepcopy(self.decoder)

        # ===
        # === this is only to be loaded by the legacy pre-trained model weight (joints-version) that
        # === are compatible with it
        self.fuser = nn.Sequential(
            nn.Conv3d(in_channels=14, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(True),
        )
        self.defuser = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv3d(in_channels=4, out_channels=14, kernel_size=5, padding=2),
            nn.ReLU(True)
        )
        # ===
        # ===

        if legacy_pretrained:
            self.load_legacy_pretrained_weights()

    def load_legacy_pretrained_weights(self):
        """
        Extract the useful weights from the VHA trained on the human pose estimation task (14 heatmaps for each joint),
        we will call it 14VHA.
        Please remember: pytorch will load weights by module names. So, for example in the 14VHA there were
        these modules: encoder, fuser, defuser, decoder
        Now for the detection, we want:
        1.  encoder and encoder_w_h (encoder_w_h to predict the width and height values, that are not guassians like in
            the "encoder", but spheres)
        2. fuser_c3d (with a kernel 3x5x5x5 instead of a 14x5x5x5 as we have seen in the "fuser")
        3. defuser_c3d (same as the fuser_c3d)
        4. decoder and decoder_w_h (in order to obtain back the heatmap and the map with the widths and heights)

        Procedure:
        1. load the 14VHA weights in the old modules (by name)
        2. transfer the encoder weights in the encoder_w_h weights using "load_state_dict"
        3. load in the fuser_c3d the weights of the fuser.
            3.1 load the same weights and biases in the second module inside the fuser_c3d
            3.2 in torch.no_grad(), load only the first 3 filters of the 14VHA in our VHA
        4. do the same for the defuser_c3d
        """
        self.load_w('log/pretrained/best.pth', strict=False, map_location=torch.device('cpu'))
        # copy the encoder and decoder weights in the new encoder and decoder used for width and height 3d-maps
        self.encoder_w_h.load_state_dict(self.encoder.state_dict(), strict=False)
        self.decoder_w_h.load_state_dict(self.decoder.state_dict(), strict=False)

        # set a subset of the legacy weights to the new reduced c3d layers
        self.fuser_c3d._modules['2'].load_state_dict(self.fuser._modules['2'].state_dict())
        self.defuser_c3d._modules['0'].load_state_dict(self.defuser._modules['0'].state_dict())
        with torch.no_grad():
            self.fuser_c3d._modules['0'].weight = nn.Parameter(self.fuser._modules['0'].weight[:, :3, :, :, :].clone())
            self.fuser_c3d._modules['0'].bias = nn.Parameter(self.fuser._modules['0'].bias.clone())
            self.defuser_c3d._modules['2'].weight = nn.Parameter(self.defuser._modules['2'].weight[:3, :, :, :, :].clone())
            self.defuser_c3d._modules['2'].bias = nn.Parameter(self.defuser._modules['2'].bias[:3].clone())

    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x.shape = batch, 3, h, w, c
        x_c = x[:, 0, :, :, :]
        x_c = x_c.squeeze(1).contiguous()
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
        x_c = x[:, 0, :, :, :]
        x_c = x_c.squeeze(1).contiguous()
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

    model = Autoencoder(legacy_pretrained=True).to(device)
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
