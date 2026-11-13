from Configure import *
import os
import torch
from albumentations import ElasticTransform, GridDistortion
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchvision import transforms as T
import re
import random
from sklearn.model_selection import KFold
import numpy as np
import cv2
import matplotlib.pyplot as plt


def sort_by_number(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(number) for number in numbers]


class DriveDataset(Dataset):
    def __init__(self, img_paths, mask_paths, train=True):
        
        self.img_paths = sorted(img_paths, key=sort_by_number)
        self.mask_paths = sorted(mask_paths, key=sort_by_number)
        self.train = train

        # self.skt_paths = skt_paths


    def __getitem__(self, index):

        image = cv2.imread(self.img_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
        if len(np.unique(mask)) == 2:
            mask[mask == 0] = 0
            mask[mask == 255] = 1
        else:
            mask[mask == 0] = 0
            mask[mask == 170] = 0
            mask[mask == 255] = 0
            mask[mask == 85] = 1
            mask[mask == 1] = 1


        if self.train:
            
            # random.seed(12)
            flipCode = random.choice([-1, 0, 1, 2, 3, 4])

            # list = [-1, 0, 1, 2]
            # i = index % 4
            # flipCode = list[i]

            # print(flipCode)
            if flipCode != -1:
                if flipCode in [0, 1, 2]:
                    rows, cols = image.shape
                    # random.seed(999)
                    t = random.randrange(0, 180, 15)
                    # l = np.arange(0, 180, 15).tolist()
                    # n = index % 12
                    # t = l[n]
                    # print(t)

                    rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), t, 1)
                    image = self.augment1(image.astype(np.float32), flipCode, rotate, 0)
                    mask = self.augment1(mask.astype(np.float32), flipCode, rotate, 1)
                elif flipCode in [3, 4]:
                    image, mask = self.augment2(image.astype(np.float32), mask.astype(np.float32), flipCode)


        max_image = image.max()
        min_image = image.min()
        image = (image - min_image) / (max_image - min_image)
        mean = 0.5
        std = 0.5
        norm_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize((mean,), (std,))
        ])
        transform = T.ToTensor()
        image = image.astype(np.float32)
        # plt.imshow(image, cmap='gray')
        
        # plt.show()

        mask = mask.astype(np.float32)
        image = norm_transform(image.copy())
        mask = transform(mask.copy())

        return image.cuda(), mask.cuda()

    def __len__(self):
        return len(self.mask_paths)

    def augment1(self, input, flipCode, rotate, flag=0):
        rows, cols = input.shape
        if flag == 0:
            flags = cv2.INTER_LINEAR
        else:
            flags = cv2.INTER_NEAREST
        if flipCode == 0:
            flip_out = np.fliplr(input)
        elif flipCode == 1:
            flip_out = cv2.warpAffine(input, rotate, (cols, rows), flags=flags)
        elif flipCode == 2:
            flip_out = np.fliplr(input)
            flip_out = cv2.warpAffine(flip_out, rotate, (cols, rows), flags=flags)
        return flip_out

    def augment2(self, input1, input2, flipCode):
        if flipCode == 3:
            image = (((input1 - input1.min()) / (input1.max() - input1.min()))*255).astype(np.uint8)
            mask = input2
            flip_out1 = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05,
                      alpha_affine=120 * 0.03)(image=image, mask=mask)
            image = flip_out1['image']
            image = image.astype(np.float32)
            mask = flip_out1['mask']

        elif flipCode == 4:
            image = (((input1 - input1.min()) / (input1.max() - input1.min()))*255).astype(np.uint8)
            mask = input2
            flip_out1 = GridDistortion(p=1, num_steps=6)(image=image, mask=mask)
            image = flip_out1['image']
            image = image.astype(np.float32)
            mask = flip_out1['mask']
        return image, mask



def load_data(phase=''):
    train_id, valid_id = get_fold(n_sub=1061, kfold=10, fold=4)
    root_dir = CFG.DATA_ROOT
    if phase == 'train':
        data_dir = os.path.join(root_dir, 'train')
        img_paths = sorted(glob(data_dir + '/temp2/Myos_img/*'))
        mask_paths = sorted(glob(data_dir + '/temp2/Myos_label/*'))

        
        img_paths_train = [img_paths[i] for i in train_id]
        mask_paths_train = [mask_paths[i] for i in train_id]
        data_set = DriveDataset(img_paths_train, mask_paths_train, train=True)
        data_loader = DataLoader(data_set, batch_size=CFG.batch_size, shuffle=True)  # true
        return data_loader

    elif phase == 'val':
        data_dir = os.path.join(root_dir, 'train')
        img_paths = sorted(glob(data_dir + '/temp2/Myos_img/*'))
        mask_paths = sorted(glob(data_dir + '/temp2/Myos_label/*'))
        img_paths_valid = [img_paths[i] for i in valid_id]
        mask_paths_valid = [mask_paths[i] for i in valid_id]
        data_set = DriveDataset(img_paths_valid, mask_paths_valid, train=False)
        data_loader = DataLoader(data_set, batch_size=CFG.batch_size, shuffle=False)
        return data_loader

    elif phase == 'test':
        data_dir = os.path.join(root_dir, 'test')
        img_paths = sorted(glob(data_dir + '/imagesTr/dcmpng2_336/*'))
        mask_paths = sorted(glob(data_dir + '/labelsTr/niipng2_336/*'))
        # skt_paths = sorted(glob(data_dir + '/mask_skt/*'))
        data_set = DriveDataset(img_paths, mask_paths)
        data_loader = DataLoader(data_set, batch_size=CFG.batch_size, shuffle=False)
        return data_loader

def get_fold(n_sub, kfold, fold):

    id = list(range(n_sub))


    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123, shuffle=True)

    train_index = list()
    valid_index = list()

    for train, valid in kf.split(np.array(id)):
        train_index.append(train)
        valid_index.append(valid)

    train_id = train_index[fold]
    valid_id = valid_index[fold]

    return train_id, valid_id


