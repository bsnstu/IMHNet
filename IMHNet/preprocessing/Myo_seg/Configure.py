import os
import torch.nn as nn


class CFG:
    # seed = 10101

    """ DIR SETTING """
    DATA_ROOT = "./preprocessing/Myo_seg/data"
    SAVE_ROOT = "./preprocessing/Myo_seg/summary"
    SAVE_ROOT_test =  "./preprocessing/Myo_seg/summary"
    # G_VIS_ROOT = './preprocessing/Myo_seg/data/gvis'

    """ TRAINING SETTING """
    batch_size = 8
    epochs = 100

    # OPT settings #
    base_lr = 0.0001
    min_lr = 1e-6
    weight_decay = 5e-7
    scheduler = 'ReduceLROnPlateau'  # CosineAnnealingLR CosineAnnealingWarmRestarts ReduceLROnPlateau

    """ SAVING SETTING """
    checkpoint_path = r"./preprocessing/Myo_seg/summary/ckpt"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)