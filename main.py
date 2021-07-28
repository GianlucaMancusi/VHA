# -*- coding: utf-8 -*-
# ---------------------

import matplotlib

matplotlib.use('Agg')

from conf import Conf

import click
import torch.backends.cudnn as cudnn
from path import Path
from termcolor import colored
from trainer_joint import TrainerJoint
from trainer_det import TrainerDet

cudnn.benchmark = True


@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, seed):
    # type: (str, int) -> None

    # if `exp_name` is None,
    # ask the user to enter it
    if exp_name is None:
        exp_name = click.prompt('▶ experiment name', default='default')

    # if `exp_name` contains a '@' character,
    # the number following '@' is considered as
    # the desired random seed for the experiment
    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(seed=seed, exp_name=exp_name)

    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    if cnf.model_input == 'joint':
        trainer = TrainerJoint(cnf=cnf)
    elif cnf.model_input == 'detection':
        trainer = TrainerDet(cnf=cnf)
    else:
        assert False, ''
    trainer.run()


if __name__ == '__main__':
    main()
