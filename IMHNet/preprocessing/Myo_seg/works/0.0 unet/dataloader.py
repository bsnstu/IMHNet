#from Configure import *
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torch_geometric.data import Data, Dataset, DataLoader
from glob import glob


class CFG:
    # seed = 10101

    """ DIR SETTING """
    DATA_ROOT = "./preprocessing/Myo_seg/data"
    SAVE_ROOT ="./preprocessing/Myo_seg/summary"
    SAVE_ROOT_test =  r"./preprocessing/Myo_seg/summary"
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

class DriveDataset(Dataset):
    def __init__(self, img_paths, mask_paths):

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # self.skt_paths = skt_paths

    def __getitem__(self, index):
        """ Reading image """
        image_filename = os.path.basename(self.img_paths[index])
        image = cv2.imread(self.img_paths[index], cv2.IMREAD_GRAYSCALE)
        print(image)
        # a = self.img_paths[index]
        # print(a)

        image = image/np.max(image)  ## (512, 512)
        # image = np.transpose(image, (2, 0, 1))  ## (1, 512, 512)
        image = np.expand_dims(image, axis=0)  ## (1, 512, 512)
        image = torch.from_numpy(image.astype(np.float32))
        print(image)

        """ Reading mask """
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
        # b = self.mask_paths[index]
        # print(b)
        # mask = mask/255.0   ## (512, 512)
        # mask[mask < 125] = 0  ## (512, 512)
        # mask[mask >= 125] = 1  ## (512, 512)
        # mask = (mask >= 75).astype(float)
        mask[mask == 0] = 0  ## (512, 512)
        mask[mask == 255] = 1
        mask = torch.from_numpy((mask).astype(np.float32)).unsqueeze(0)

        # """ Reading skt """
        # skt = cv2.imread(self.skt_paths[index], cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        # skt = torch.from_numpy((skt).astype(np.float32)).unsqueeze(0)
        return image.cuda(), mask.cuda(), image_filename

    def __len__(self):
        return len(self.mask_paths)


def load_data(phase=''):

    root_dir = CFG.DATA_ROOT
    if phase == 'train':
        data_dir = os.path.join(root_dir, 'train')
        img_paths = sorted(glob(data_dir + '/images/*'))
        mask_paths = sorted(glob(data_dir + '/labels/*'))
        # skt_paths = sorted(glob(data_dir + '/mask_skt/*'))
        data_set = DriveDataset(img_paths, mask_paths)
        data_loader = DataLoader(data_set, batch_size=CFG.batch_size, shuffle=True)  # true
        return data_loader

    elif phase == 'val':
        data_dir = os.path.join(root_dir, 'valid')
        img_paths = sorted(glob(data_dir + '/images/*'))
        mask_paths = sorted(glob(data_dir + '/labels/*'))
        # skt_paths = sorted(glob(data_dir + '/mask_skt/*'))
        data_set = DriveDataset(img_paths, mask_paths)
        data_loader = DataLoader(data_set, batch_size=CFG.batch_size, shuffle=True)
        return data_loader

    elif phase == 'test':
        data_dir = os.path.join(root_dir, 'test')

        img_paths = sorted(glob(data_dir + '/total_data/*'))
        #print(img_paths)
        mask_paths = sorted(glob(data_dir + '/total_label/*'))
        print("img_paths num:", len(img_paths))
        print("mask_paths num:", len(mask_paths))
        data_set = DriveDataset(img_paths, mask_paths)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
        return data_loader

