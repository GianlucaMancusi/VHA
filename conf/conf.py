# -*- coding: utf-8 -*-
# ---------------------


import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()

    def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True):
        # type: (str, int, str, bool) -> None
        """
        :param conf_file_path: optional path of the configuration file
        :param seed: desired seed for the RNG; if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param log: `True` if you want to log each step; `False` otherwise
        """
        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        self.project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_log_path = Path('log')
        self.exp_log_path = self.project_log_path / exp_name

        # set random seed
        self.seed = set_seed(seed)  # type: int

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuation file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.FullLoader)

        # read configuration parameters from YAML file
        # or set their default value
        self.model_input = str(y.get('MODEL_INPUT', 'joint'))  # type: str
        self.detection_model = str(y.get('DETECTION_MODEL', 'c2d-shared'))  # type: str
        self.loss_function = str(y.get('LOSS_FUNCTION', 'MSE'))  # type: str
        self.q = y.get('Q', 0.31746031746031744)  # type: float # --> quantization factor
        self.hmap_h = y.get('H', 128)  # type: int
        self.hmap_w = y.get('W', 128)  # type: int
        self.hmap_d = y.get('D', 100)  # type: int
        self.sigma = y.get('SIGMA', 2)  # type: int
        self.sphere_diameter = y.get('SPHERE_DIAMETER', 5)  # type: int
        self.lr = y.get('LR', 0.0001)  # type: float
        self.epochs = y.get('EPOCHS', 10000)  # type: int
        self.n_workers = y.get('N_WORKERS', 8)  # type: int
        self.batch_size = y.get('BATCH_SIZE', 1)  # type: int
        self.test_len = y.get('TEST_LEN', 128)  # type: int
        self.epoch_len = y.get('EPOCH_LEN', 1024)  # type: int
        self.data_augmentation = str(y.get('DATA_AUGMENTATION', 'no'))  # type: str # --> 'no': no data aug, 'images': only images, 'all', images and heatmap
        self.mot_synth_ann_path = y.get('MOTSYNTH_ANN_PATH', '. / motsynth')  # type: str
        self.mot_synth_path = y.get('MOTSYNTH_PATH', '. / motsynth')  # type: str

        if y.get('DEVICE', None) is not None and y['DEVICE'] != 'cpu':
            #os.environ['CUDA_VISIBLE_DEVICES'] = str(y.get('DEVICE').split(':')[1])
            self.device = 'cuda'
        elif y.get('DEVICE', None) is not None and y['DEVICE'] == 'cpu':
            self.device = 'cpu'
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        available_data_aug = ['no', 'images', 'all']
        assert self.data_augmentation in available_data_aug, f'the specified DATA_AUGMENTATION parameter "{self.data_augmentation}" does not exist, it must be one of {available_data_aug}'

        available_model_input = ['joint', 'detection', 'tracking']
        assert self.model_input in available_model_input, f'the specified MODEL_INPUT parameter "{self.model_input}" does not exist, it must be one of {available_model_input}'

        available_detection_model = ['c2d-shared', 'c2d-divided', 'c2d-divided-c3d-pretrained']
        assert self.detection_model in available_detection_model, f'the specified DETECTION_MODEL parameter "{self.detection_model}" does not exist, it must be one of {available_detection_model}'

        available_loss_functions = ['MSE', 'L1']
        assert self.loss_function in available_loss_functions, f'the specified LOSS_FUNCTION parameter "{self.loss_function}" does not exist, it must be one of {available_loss_functions}'

        self.mot_synth_ann_path = Path(self.mot_synth_ann_path)
        self.mot_synth_path = Path(self.mot_synth_path)
        assert self.mot_synth_ann_path.exists(), 'the specified directory for the MOTSynth-Dataset does not exists'
        assert self.mot_synth_path.exists(), 'the specified directory for the MOTSynth-Dataset does not exists'

        self.saved_epoch = 0
        self.model_weights = None
        self.best_val_f1 = None
        self.optimizer_data = None

        # loading checkpoint data
        ck_path = self.exp_log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=torch.device('cpu'))
            print('[loading checkpoint \'{}\']'.format(ck_path))
            self.saved_epoch = ck['epoch']
            self.model_weights = ck['model']
            self.best_val_f1 = ck['best_val_f1']
            if ck.get('optimizer', None) is not None:
                self.optimizer_data = ck['optimizer']
