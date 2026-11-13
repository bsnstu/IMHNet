import torch

from Configure import *
import numpy as np
import cv2


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


def visualization_segmentation(img, pred, mask, name, phase=''):
    """ Saving masks """
    # pred = torch.sigmoid(pred)
    print("Hello")
    img = img.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    pred = (pred >= 0.5).astype(np.float32)
    line = np.ones((256, 1)) * 128

    """ test """
    cat_images = np.concatenate(
        [img * 255, line, line, pred * 255], axis=1
    )

    """ train """
    # cat_images = np.concatenate(
    #     [img * 255, line, mask * 255, line, pred * 255], axis=1
    # )
    if phase == 'train':
        cv2.imwrite(CFG.SAVE_ROOT + '/seg_tra/{}.png'.format(name), cat_images)

    elif phase == 'valid':
        cv2.imwrite(CFG.SAVE_ROOT + '/seg_val/{}.png'.format(name), cat_images)

    elif phase == 'test':
       # cv2.imwrite(CFG.SAVE_ROOT_test + '/temp3/MS/Train/seg_val_2242/{}.png'.format(name), cat_images)
       # cv2.imwrite(CFG.SAVE_ROOT_test + '/temp3/MS/Train/Val_img_2242/{}.png'.format(name), img * 255)
        cv2.imwrite(CFG.SAVE_ROOT_test + '/delete_img/{}.png'.format(name),pred*255)
        print("Hello")