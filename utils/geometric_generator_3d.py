import torch


class GeometricGenerator3D:

    def __init__(self, cnf):
        self.cnf = cnf

    def paste_tensor(self, x, tensor_to_paste, size, center, mul_value=1, use_max=True):
        center = center[::-1]

        xa, ya, za = max(0, center[2] - size // 2), max(0, center[1] - size // 2), max(0, center[
            0] - size // 2)
        xb, yb, zb = min(center[2] + size // 2, self.cnf.hmap_w - 1), min(center[1] + size // 2,
                                                                     self.cnf.hmap_h - 1), min(
            center[0] + size // 2, self.cnf.hmap_d - 1)
        hg, wg, dg = (yb - ya) + 1, (xb - xa) + 1, (zb - za) + 1

        gxa, gya, gza = 0, 0, 0
        gxb, gyb, gzb = size - 1, size - 1, size - 1

        if center[2] - size // 2 < 0:
            gxa = -(center[2] - size // 2)
        if center[1] - size // 2 < 0:
            gya = -(center[1] - size // 2)
        if center[0] - size // 2 < 0:
            gza = -(center[0] - size // 2)
        if center[2] + size // 2 > (self.cnf.hmap_w - 1):
            gxb = wg - 1
        if center[1] + size // 2 > (self.cnf.hmap_h - 1):
            gyb = hg - 1
        if center[0] + size // 2 > (self.cnf.hmap_d - 1):
            gzb = dg - 1

        if use_max:
            x[za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(
                torch.cat(tuple([
                    x[za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0),
                    tensor_to_paste[gza:gzb + 1, gya:gyb + 1, gxa:gxb + 1].unsqueeze(0) * mul_value
                ])), 0)[0]
        else:
            x[za:zb + 1, ya:yb + 1, xa:xb + 1] = tensor_to_paste[gza:gzb + 1, gya:gyb + 1, gxa:gxb + 1] * mul_value

    def gen_3d_spheres(self, positions, values):
        pass

    @staticmethod
    def make_a_gaussian(d, h, w, center, s=2, device='cpu'):
        # type: (int, int, int, Union[List[int], Tuple[int, int, int]], float, str) -> torch.Tensor
        """
        :param d: hmap depth
        :param h: hmap height
        :param w: hmap width
        :param center: center of the Gaussian | ORDER: (x, y, z)
        :param s: sigma of the Gaussian
        :param device:
        :return: heatmap (shape torch.Size([d, h, w])) with a gaussian centered in `center`
        """
        x = torch.arange(0, w, 1).float().to(device)
        y = torch.arange(0, h, 1).float().to(device)
        y = y.unsqueeze(1)
        z = torch.arange(0, d, 1).float().to(device)
        z = z.unsqueeze(1).unsqueeze(1)

        x0 = center[0]
        y0 = center[1]
        z0 = center[2]

        return torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / s ** 2)

    @staticmethod
    def make_a_sphere(d, h, w, center, r=2, device='cpu'):
        # type: (int, int, int, Union[List[int], Tuple[int, int, int]], float, str) -> torch.Tensor
        """
        :param d: hmap depth
        :param h: hmap height
        :param w: hmap width
        :param center: center of the Gaussian | ORDER: (x, y, z)
        :param r: radius of the Sphere
        :param device:
        :return: heatmap (shape torch.Size([d, h, w])) with a sphere centered in `center`
        """
        x = torch.arange(0, w, 1).float().to(device)
        y = torch.arange(0, h, 1).float().to(device)
        y = y.unsqueeze(1)
        z = torch.arange(0, d, 1).float().to(device)
        z = z.unsqueeze(1).unsqueeze(1)

        x0 = center[0]
        y0 = center[1]
        z0 = center[2]

        return torch.where((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 <= r ** 2, 1, 0)
