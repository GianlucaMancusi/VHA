
def main():
    MAX_WIDTH = 1919
    MAX_HEIGHT = 1079
    from test_metrics import joint_det_metrics, compute_det_metrics_iou
    import json
    import numpy as np
    from torch.utils.data import DataLoader
    from conf import Conf
    from dataset.mot_synth_det_ds import MOTSynthDetDS
    from utils import utils
    import torch

    cnf = Conf(exp_name='vha_d_debug', preload_checkpoint=False)

    # load dataset
    mode = 'test'
    ds = MOTSynthDetDS(mode=mode, cnf=cnf)
    loader = DataLoader(dataset=ds, batch_size=1, num_workers=0, shuffle=False)

    # load model
    from models.vha_det_variable_versions import Autoencoder as AutoencoderVariableVersions
    model = AutoencoderVariableVersions(vha_version=1).to(cnf.device)
    model.eval()
    model.requires_grad(False)
    if cnf.model_weights is not None:
        model.load_state_dict(torch.load(cnf.exp_log_path / 'best.pth', map_location=torch.device('cpu')), strict=False)

    # ======== MAIN LOOP ========
    for i, sample in enumerate(loader):
        x, y, file_name, aug_info = None, None, None, None

        if mode == 'test':
            x, y, file_name, aug_info = sample
            y_true = json.loads(y[0])
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
        y_center_pred = utils.local_maxima_3d(heatmap=x_pred_center, threshold=0.1, device=cnf.device)
        y_width_pred = []
        y_height_pred = []
        bboxes_info_pred = []
        # w_min = min([float(x_width[cam_dist, y2d, x2d]) for cam_dist, y2d, x2d in y_center_pred])
        # w_max = max([float(x_width[cam_dist, y2d, x2d]) for cam_dist, y2d, x2d in y_center_pred])
        # h_min = min([float(x_height[cam_dist, y2d, x2d]) for cam_dist, y2d, x2d in y_center_pred])
        # h_max = max([float(x_height[cam_dist, y2d, x2d]) for cam_dist, y2d, x2d in y_center_pred])
        for cam_dist, y2d, x2d in y_center:
            width = float(x_pred_width[cam_dist, y2d, x2d])
            height = float(x_pred_height[cam_dist, y2d, x2d])

            # denormalize width and height
            width = int(round(width * MAX_WIDTH))
            height = int(round(height * MAX_HEIGHT))

            y_width_pred.append((cam_dist, y2d, x2d, width))
            y_height_pred.append((cam_dist, y2d, x2d, height))

            x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=cnf.q)
            bboxes_info_pred.append((x2d - width / 2, y2d - height / 2, width, height, cam_dist))

        img_original = np.array(utils.imread(cnf.mot_synth_path / file_name[0]).convert("RGB"))
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
            print(f'f1_iou={f1_iou}, f1_center={f1_center}, f1_width={f1_width}, f1_height={f1_height}')

            # for cam_dist, y2d, x2d, width, height in y_true:
            #    x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=cnf.q)
            #    bboxes_info_true.append((x2d - width / 2, y2d - height / 2, width, height, cam_dist))

            # utils.visualize_bboxes(img_original, bboxes_info_true, use_z=True, half_images=False, aug_info=aug_info,
            #                       normalize_z=False)

        # print(f'({i}) Dataset example: x.shape={tuple(x.shape)}, y={y}')

        utils.visualize_bboxes(img_original, bboxes_info_pred, use_z=True, half_images=True, aug_info=aug_info,
                               normalize_z=False)

if __name__ == '__main__':
    main()