# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import PIL
import numpy as np
import torch
from PIL.Image import Image
from path import Path
import cv2
from colour import Color

from conf import Conf


MAX_LS = np.array([0.27, 0.41, 0.67, 0.93, 0.41, 0.67, 0.92, 0.88, 1.28, 1.69, 0.88, 1.29, 1.70])


def imread(path):
    # type: (Union[Path, str]) -> Image
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def torch_to_list(y):
    # type: (torch.Tensor) -> List[List[int]]
    if len(y.shape) == 3:
        y = y[0]
    return [list(point3d.cpu().numpy()) for point3d in y]


def local_maxima_3d(heatmap, threshold, device='cuda', ret_confs=False):
    # type: (torch.Tensor, float, str, bool) -> Union[List[List[int]], Tuple[List[List[int]], List[float]]]
    """
    :param heatmap: 3D heatmap with shape (D, H, W)
    :param threshold: peaks with values < of that threshold will not be returned
    :param device: device where you want to run the operation
    :param ret_confs: do you want to return confidence values (one for each peak)?
    :return: list of detected peak(s) and (optional) confidence values
    """
    d = torch.device(device)

    confidences = []
    m_f = torch.zeros(heatmap.shape).to(d)
    m_f[1:, :, :] = heatmap[:-1, :, :]
    m_b = torch.zeros(heatmap.shape).to(d)
    m_b[:-1, :, :] = heatmap[1:, :, :]

    m_u = torch.zeros(heatmap.shape).to(d)
    m_u[:, 1:, :] = heatmap[:, :-1, :]
    m_d = torch.zeros(heatmap.shape).to(d)
    m_d[:, :-1, :] = heatmap[:, 1:, :]

    m_r = torch.zeros(heatmap.shape).to(d)
    m_r[:, :, 1:] = heatmap[:, :, :-1]
    m_l = torch.zeros(heatmap.shape).to(d)
    m_l[:, :, :-1] = heatmap[:, :, 1:]

    p = torch.zeros(heatmap.shape).to(d)
    p[heatmap >= m_f] = 1
    p[heatmap >= m_b] += 1
    p[heatmap >= m_u] += 1
    p[heatmap >= m_d] += 1
    p[heatmap >= m_r] += 1
    p[heatmap >= m_l] += 1

    p[heatmap >= threshold] += 1
    p[p != 7] = 0

    peaks = torch.nonzero(p).cpu()
    peaks = [[z, y, x] for z, y, x in torch_to_list(peaks)]

    if ret_confs:
        tmp_c = torch.nonzero(p).cpu()
        tmp_c = [heatmap[z, y, x].item() for z, y, x in torch_to_list(tmp_c)]
        confidences += tmp_c

    if ret_confs:
        return peaks, confidences
    else:
        return peaks


def get_3d_hmap_image(cnf, hmap, image, coords2d, normalize=False, scale_to=None, image_weight=None):
    """
    A 2d image that shows the depth in a gradient color
    :param hmap: 3D heatmap with values in [0,1] and shape (D, H, W)
    :param image: 2d image (3, H, W)
    """

    if coords2d is None:
        coords2d = []

    if scale_to is None:
        scale_to = (320, 180)

    if image_weight is None:
        image_weight = .8
        if coords2d is None:
            image_weight = .5

    if not (type(hmap) is np.ndarray):
        try:
            hmap = hmap.cpu().numpy()
        except:
            hmap = hmap.detach().cpu().numpy()

    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    if normalize:
        hmap = hmap / hmap.max()
    hmap = (hmap * 255).astype(np.uint8)
    collpsed_hmap_in_2d = np.max(hmap, axis=(0, 1))
    collpsed_hmap_in_2d = cv2.applyColorMap(collpsed_hmap_in_2d, colormap=cv2.COLORMAP_JET)
    image = cv2.resize(image, scale_to)
    collpsed_hmap_in_2d = cv2.resize(collpsed_hmap_in_2d, scale_to, interpolation=cv2.INTER_NEAREST)
    final_image = cv2.addWeighted(collpsed_hmap_in_2d, (1 - image_weight), image, image_weight, 0.0)

    # distance representation by using dots:
    if len(coords2d) > 0:
        min_d = min([coord[1] for coord in coords2d])
        max_d = max([coord[1] for coord in coords2d])
        range_d = int(max_d - min_d)
        colors = list(Color("green").range_to(Color("red"), range_d + 1))
        for _, d, y, x in coords2d:
            d_index = int(d - min_d)
            y_ratio, x_ratio = scale_to[1] / cnf.hmap_h, scale_to[0] / cnf.hmap_w
            color = [int(c * 255) for c in colors[d_index].rgb]
            final_image = cv2.drawMarker(final_image, (int(x * x_ratio), int(y * y_ratio)),
                                         (color[2], color[1], color[0]),
                                         markerType=cv2.MARKER_DIAMOND, markerSize=2, thickness=1,
                                         line_type=cv2.LINE_AA)
    # final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    return final_image


def draw_bboxes(image, bboxes, use_z=False, half_images=True, aug_info=None, normalize_z=True):
    image = image.copy()
    image = cv2.resize(image, (1920 // 2, 1080 // 2) if half_images else (1920, 1080))
    if aug_info is not None:
        import imgaug.augmenters as iaa
        aug_scale, aug_h, aug_w = aug_info
        aug_scale, aug_h, aug_w = float(aug_scale), float(aug_h), float(aug_w)
        img_h, img_w, _ = image.shape
        # convert the offset calculated for 3d points (for the 3d heat map) to offset useful for
        # the Affine transformation by using the imgaug library
        aug_offset_h = -(aug_h - .5) * (img_h * aug_scale - img_h)
        aug_offset_w = -(aug_w - .5) * (img_w * aug_scale - img_w)
        aug_affine = iaa.Affine(scale=aug_scale,
                                translate_px={'x': int(round(aug_offset_w)), 'y': int(round(aug_offset_h))})
        image = aug_affine(image=image, return_batch=False)
    if len(bboxes) == 0:
        return image
    if use_z:
        min_d = int(round(min([b[4] for b in bboxes])))
        max_d = int(round(max([b[4] for b in bboxes])))
        if normalize_z:
            range_d = max_d - min_d
            colors = list(Color("green").range_to(Color("red"), range_d + 1))
        else:
            colors = list(Color("green").range_to(Color("red"), 48))
            colors += list(Color("red").range_to(Color("black"), 140 - 48))
            colors += list(Color("black").range_to(Color("violet"), 316 - 140 - 48))

        for x, y, w, h, d in bboxes:
            if half_images:
                x, y, w, h = x / 2, y / 2, w / 2, h / 2
            d = int(round(d))
            if d > 58:
                print("",end="")
            d_index = d - min_d if normalize_z else d
            color = [int(round(c * 255)) for c in colors[d_index].rgb]
            image = cv2.rectangle(image, (int(x), int(y)),
                                  (int(x) + int(w), int(y) + int(h)),
                                  color=(color[0], color[1], color[2]), thickness=2)
            image = cv2.putText(image, str(d), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (color[2], color[1], color[0]), 1, cv2.LINE_AA)
    else:

        for bbox in bboxes:
            image = cv2.rectangle(image, (int(bbox[0]) - int(bbox[2] / 2), int(bbox[1]) - int(bbox[3] / 2)),
                                  (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])), color=(255, 0, 0),
                                  thickness=2)
    return image


def visualize_bboxes(image, bboxes, use_z=False, half_images=True, aug_info=None, normalize_z=True):
    x = draw_bboxes(image, bboxes, use_z, half_images, aug_info, normalize_z)
    cv2.imshow(f'press ESC to exit', cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyAllWindows()


def save_bboxes(image, bboxes, path, use_z=False, half_images=True, aug_info=None, normalize_z=True):
    x = draw_bboxes(image, bboxes, use_z, half_images, aug_info, normalize_z)
    cv2.imwrite(path, cv2.cvtColor(x, cv2.COLOR_RGB2BGR))


def visualize_3d_hmap(hmap, image=None, depth_limit=315):
    # type: (Union[np.ndarray, torch.Tensor]) -> None
    """
    Interactive visualization of 3D heatmaps.
    :param hmap: 3D heatmap with values in [0,1] and shape (D, H, W)
    """

    if not (type(hmap) is np.ndarray):
        try:
            hmap = hmap.cpu().numpy()
        except:
            hmap = hmap.detach().cpu().numpy()

    hmap = (hmap + hmap.min()) / (hmap.max() + hmap.min())
    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    hmap = (hmap * 255).astype(np.uint8)
    for d, x in enumerate(hmap):
        if d > depth_limit:
            break

        x = cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET)

        if image is not None:
            image = cv2.resize(image, (1280, 720))
            x = cv2.resize(x, (1280, 720), interpolation=cv2.INTER_NEAREST)
            x = cv2.addWeighted(x, .5, image, .5, 0.0)

        x = cv2.putText(x, f'{d}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 128, 128), 2, cv2.LINE_AA)
        cv2.imshow(f'press ESC to advance in the depth dimension', x)
        cv2.waitKey()
    cv2.destroyAllWindows()


def visualize_multiple_3d_hmap(hmaps):
    # type: (List[torch.Tensor], Any) -> None
    """
    Interactive visualization of 3D heatmaps.
    :param hmap: 3D heatmap with values in [0,1] and shape (D, H, W)
    """
    import cv2

    hmap_gt = hmaps[0].cpu().numpy()
    hmap_pr = hmaps[1].cpu().numpy()

    hmap = np.concatenate((hmap_gt, hmap_pr), axis=2)
    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    hmap = (hmap * 255).astype(np.uint8)
    for d, x in enumerate(hmap):
        x = cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET)
        x = cv2.putText(x, f'{d}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 128, 128), 2, cv2.LINE_AA)
        cv2.imshow(f'press ESC to advance in the depth dimension', x)
        cv2.waitKey()
    cv2.destroyAllWindows()


def gkern(d, h, w, center, s=2, device='cuda'):
    # type: (int, int, int, Union[List[int], Tuple[int, int, int]], float, str) -> torch.Tensor
    """
    :param d: hmap depth
    :param h: hmap height
    :param w: hmap width
    :param center: center of the Gaussian | ORDER: (x, y, z)
    :param s: sigma of the Gaussian
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


def rescale_to_real(x2d, y2d, cam_dist, q):
    # type: (int, int, float, float) -> Tuple[int, int, float]
    """
    :param x2d: predicted `x2d` (real_x2d // 8)
    :param y2d: predicted `y2d` (real_y2d // 8)
    :param cam_dist: predicted `cam_distance` (real_cam_distance // 0.317)
    :param q: quantization coefficient
    :return: (real_x2d, real_y2d, real_cam_distance)
    """
    return x2d * 8, y2d * 8, (cam_dist * q)


def to3d(x2d, y2d, cam_dist, fx, fy, cx, cy):
    # type: (int, int, float, float, float, float, float) -> Tuple[float, float, float]
    """
    Converts a 2D point on the image plane into a 3D point in the standard
    coordinate system using the intrinsic camera parameters.

    :param x2d: 2D x coordinate [px]
    :param y2d: 2D y coordinate [px]
    :param cam_dist: distance from camera [m]
    :param fx: x component of the focal len
    :param fy: y component of the focal len
    :param cx: x component of the central point
    :param cy: y component of the central point
    :return: 3D coordinates
    """

    k = (-1) * np.sqrt(
        fx ** 2 * fy ** 2 + fx ** 2 * cy ** 2 - 2 * fx ** 2 * cy * y2d + fx ** 2 * y2d ** 2 +
        fy ** 2 * cx ** 2 - 2 * fy ** 2 * cx * x2d + fy ** 2 * x2d ** 2
    )

    x3d = ((fy * cam_dist * cx) - (fy * cam_dist * x2d)) / k
    y3d = ((fx * cy * cam_dist) - (fx * cam_dist * y2d)) / k
    z3d = -(fx * fy * cam_dist) / k

    return x3d, y3d, z3d


def to2d(points_nx3, fx, fy, cx, cy, return_z=False):
    # type: (List[Union[np.ndarray, np.ndarray]], float, float, float, float) -> Union[Any, np.ndarray, np.ndarray]
    """
    Converts a 3D point on the image plane into a 2D point in the standard
    coordinate system using the intrinsic camera parameters.

    :param points_nx3: 3D [(x,y,z),...] coordinates
    :param fx: x component of the focal len
    :param fy: y component of the focal len
    :param cx: x component of the central point
    :param cy: y component of the central point
    :return: 2D coordinates
    """

    points_2d, _ = cv2.projectPoints(np.array(points_nx3), (0, 0, 0), (0, 0, 0), np.array([[fx, 0, cx],
                                                                                           [0, fy, cy],
                                                                                           [0, 0, 1]],
                                                                                          dtype=np.float32),
                                     np.array([]))
    if not return_z:
        return np.squeeze(points_2d, axis=1)
    else:
        return np.squeeze(points_2d, axis=1), np.array(points_nx3).transpose(1, 0)[2].mean()


def to2d_by_def(points_nx3, fx, fy, cx, cy):
    # type: (int, int, float, float, float, float, float) -> Tuple[float, float, float]
    """
    Converts a 3D point on the image plane into a 2D point in the standard
    coordinate system using the intrinsic camera parameters.

    :param points_nx3: 3D [(x,y,z),...] coordinates
    :param fx: x component of the focal len
    :param fy: y component of the focal len
    :param cx: x component of the central point
    :param cy: y component of the central point
    :return: 2D coordinates
    """
    points_2d = []
    for point in points_nx3:
        x3d, y3d, z3d = point
        x2d = fx * x3d / z3d + cx
        y2d = fy * y3d / z3d + cy
        points_2d.append([x2d, y2d])
    return points_2d


def normalize_rr_pose(rr_pose, max_ls=MAX_LS):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    :param rr_pose: root relative pose
    :param max_ls: normalization array
    :return: normalized version of the input `rr_pose`
    """
    for i in range(3):
        rr_pose[:, i] = 0.5 + rr_pose[:, i] / (2 * max_ls)
    return rr_pose


def denormalize_rr_pose(rr_pose, max_ls=MAX_LS):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    :param rr_pose: normalized root relative pose
    :param max_ls: normalization array
    :return: denormalized version of the input `rr_pose`
    """
    for i in range(3):
        rr_pose[:, i] = (rr_pose[:, i] - 0.5) * 2 * max_ls
    return rr_pose


def get_multi_local_maxima_3d(hmaps3d, threshold, device='cuda'):
    # type: (torch.Tensor, float, str) -> List[Tuple[int, int, int, int]]
    """
    :param hmaps3d: 3D heatmaps with shape (N_joints, D, H, W)
    :param threshold: peaks with values < of that threshold will not be returned
    :param device: device where you want to run the operation
    :return: ...
    """
    d = torch.device(device)

    peaks = []

    for jtype, hmap3d in enumerate(hmaps3d):
        m_f = torch.zeros(hmap3d.shape).to(d)
        m_f[1:, :, :] = hmap3d[:-1, :, :]
        m_b = torch.zeros(hmap3d.shape).to(d)
        m_b[:-1, :, :] = hmap3d[1:, :, :]

        m_u = torch.zeros(hmap3d.shape).to(d)
        m_u[:, 1:, :] = hmap3d[:, :-1, :]
        m_d = torch.zeros(hmap3d.shape).to(d)
        m_d[:, :-1, :] = hmap3d[:, 1:, :]

        m_r = torch.zeros(hmap3d.shape).to(d)
        m_r[:, :, 1:] = hmap3d[:, :, :-1]
        m_l = torch.zeros(hmap3d.shape).to(d)
        m_l[:, :, :-1] = hmap3d[:, :, 1:]

        p = torch.zeros(hmap3d.shape).to(d)
        p[hmap3d >= m_f] = 1
        p[hmap3d >= m_b] += 1
        p[hmap3d >= m_u] += 1
        p[hmap3d >= m_d] += 1
        p[hmap3d >= m_r] += 1
        p[hmap3d >= m_l] += 1

        p[hmap3d >= threshold] += 1
        p[p != 7] = 0

        tmp = torch.tensor(torch.nonzero(p).cpu())
        tmp = [[jtype, z, y, x] for z, y, x in torch_to_list(tmp)]
        peaks += tmp

    return peaks


def save_3d_hmap(hmap, path, shift_values=False):
    # type: (Union[np.ndarray, torch.Tensor], str) -> None
    """
    Saves a 3D heatmap as MP4 video with JET colormap.
    :param hmap: 3D heatmap with values in [0,1] and shape (D, H, W)
    :param path: desired path for the output video
    """
    import cv2
    import imageio

    if not (type(hmap) is np.ndarray):
        try:
            hmap = hmap.cpu().numpy()
        except:
            hmap = hmap.detach().cpu().numpy()

    if shift_values:
        shift_amount = hmap.max() - hmap.min() + 3
        hmap[hmap == shift_amount] = 0
        hmap = hmap / hmap.max()
    else:
        hmap[hmap < 0] = 0
        hmap[hmap > 1] = 1

    hmap = (hmap * 255).astype(np.uint8)
    frames = [cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET) for x in hmap]

    for d, x in enumerate(frames):
        frames[d] = cv2.putText(frames[d], f'{d}', (10, 60), 1, 2, (255, 128, 128), 2, cv2.LINE_AA)[:, :, ::-1]

    imageio.mimsave(path, frames, macro_block_size=None)
