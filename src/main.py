import os
import sys
sys.path.append('./src')

import argparse
import random
import numpy as np
from utils.configure import get_configure
from utils.logger import get_logger
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from model.train import Train
from data.data import DataManager


def set_env(seed):
    random.seed(seed)
    np.random.seed(seed)


def fold_check(config):
    log_dir = Path(config.log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    output_dir = Path(config.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tensorboard_dir = Path(config.tensorboard_dir)
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    checkpoints_dir = Path(config.checkpoints_dir)
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='conf/system.yaml', help='tuning unet on ..')
    args = parser.parse_args()

    config = get_configure(args.conf)
    fold_check(config)

    logger = get_logger(config.log_dir)
    config.show_data_summary(logger)
    set_env(config.seed)

    data_manager = DataManager(config)
    train = Train(config, logger, data_manager)
    if config.mode == 'train':
        logger.info(f'Training')
        train.train()

    else:
        logger.info(f"Testing")


        
