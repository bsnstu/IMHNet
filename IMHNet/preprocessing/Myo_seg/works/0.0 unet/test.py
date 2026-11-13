"""测试"""

import os
import time
import torch
from dataloader import load_data
from model import network
#from runner.metrics import calculate_metrics
import numpy as np
import random
import csv
import sys
from pathlib import Path
current_file = Path(__file__).resolve()


project_root = Path("./preprocessing/Myo_seg")


sys.path.append(str(project_root))
from runner.visualization import visualization_segmentation as vis
import warnings
warnings.filterwarnings("ignore")


""" Seeding the randomness. """

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    """ Seeding """
    seeding(10001)

    test_loader = load_data(phase='test')

    
   # models_dir = r"./preprocessing/Myo_seg/summary/vm"

    
    # result_csv_path = "./preprocessing/Myo_seg/summary/test_results.csv"
    # with open(result_csv_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Jaccard'])

    """ Load and test each model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_taken = []
    # metrics_dict = {'CAL': [], 'MET': []}

    
   # model_files = os.listdir(models_dir)

    # model_path = os.path.join(models_dir, model_file)
    #model_path = r"./preprocessing/Myo_seg/summary/ckpt/Myo_new1.pth"
    model_path="./preprocessing/Myo_seg/summary/ckpt/Myo_new1.pth"
    print(model_path)
    
    model = network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        for step, samples in enumerate(test_loader):
            image, mask, file_name = samples
            file_name = file_name[0].replace('.png', '')
            start_time = time.time()
            pred = model(image)
            total_time = time.time() - start_time
            time_taken.append(total_time)
            for i in range(image.size(0)):
                vis(image[i, 0, ...], torch.sigmoid(pred.data)[i, 0, ...], mask[i, 0, ...], '{}'.format(file_name), phase='test')

        # cal = np.vstack(metrics_dict['CAL']).mean(axis=0)
        # met = np.vstack(metrics_dict['MET']).mean(axis=0)

        # precision, recall, f1, acc, jaccard = met[0], met[1], met[2], met[3], met[4]
        # print(f"Model: ckpt- Precision: {precision:1.4f} - Recall: {recall:1.4f} - F1-Score: {f1:1.4f} - Accuracy: {acc:1.4f} - Jaccard: {jaccard:1.4f} ")

        
        # writer.writerow([precision, recall, f1, acc, jaccard])




# from Configure import *
# import os
# import time
# # from glob import glob
# # from operator import add
# import torch
# from tqdm import tqdm
# from dataloader import load_data
# from model import network
# # from runner.losses import DiceLoss, DiceBCELoss
# from runner.utils import seeding, create_dir, epoch_time
# # from test import calculate_metrics, mask_parse
# from runner.losses import criterion
# from runner.metrics import calculate_metrics
# import numpy as np
# import pandas as pd
# import random
# from glob import glob
# import cv2
# from runner.visualization import visualization_segmentation as vss
#
#
# """ Seeding the randomness. """
# def seeding(seed):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# if __name__ == "__main__":
#     """ Seeding """
#     seeding(10001)
#
#     """ Folders """
#     # create_dir("results")
#
#     """ Load dataset """
#     data_loader = load_data(phase='test')
#
#     """ Hyperparameters """
#     checkpoint_path = CFG.checkpoint_path
#
#     """ Load the checkpoint """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     """ DATASET """
#     root_dir = CFG.DATA_ROOT
#     data_dir = os.path.join(root_dir, 'test')
#     img_paths = sorted(glob(data_dir + '/images/*'))
#     mask_paths = sorted(glob(data_dir + '/labels/*'))
#     # skt_paths = sorted(glob(data_dir + '/mask_skt/*'))
#
#     model = network().cuda()
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     model.eval()
#
#     time_taken = []
#
#     epoch_loss = 0.0
#     metrics_dict = {'CAL': [], 'MET': []}
#
#     with torch.no_grad():
#         for idx, (img_path, mask_path) in tqdm(enumerate(zip(img_paths, mask_paths)), total=len(img_paths)):
#             """ Extract the name """
#             name = img_path.split("\\")[-1].split(".")[0]
#
#             """ Reading image """
#             image = cv2.imread(img_path, cv2.IMREAD_COLOR)
#             image = image / 255.0  # (512, 512, 3)
#             img_vis = image
#             image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
#             image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#
#             """ Reading mask """
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             # mask = mask/255.0   # (512, 512)
#             mask[mask < 125] = 0  # (512, 512)
#             mask[mask >= 125] = 1  # (512, 512)
#             mask_vis = mask
#             mask = torch.from_numpy((mask).astype(np.float32)).unsqueeze(0).unsqueeze(0)
#
#             """ Reading skt """
#             # skt = cv2.imread(skt_path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
#             # skt = torch.from_numpy((skt).astype(np.float32)).unsqueeze(0).unsqueeze(0)
#
#             start_time = time.time()
#
#             pred = model(image.cuda())
#
#             total_time = time.time() - start_time
#             time_taken.append(total_time)
#
#             pred_vis = (pred.detach().cpu().numpy()[0, 0, ...] > 0.5).astype(np.float32)
#
#             metrics = calculate_metrics(torch.sigmoid(pred.data).cpu().numpy(), mask.data.cpu().numpy())
#             metrics_dict['CAL'].append(metrics['CAL']['value'])
#             metrics_dict['MET'].append(metrics['MET']['value'])
#
#             vss(img_vis, mask_vis, pred_vis, name)
#
#         cal = np.vstack(metrics_dict['CAL']).mean(axis=0)
#         met = np.vstack(metrics_dict['MET']).mean(axis=0)
#
#     precision, recall, f1, acc, jaccard = met[0], met[1], met[2], met[3], met[4]
#     print(f"Precision: {precision:1.4f} - Recall: {recall:1.4f} - F1-Score: {f1:1.4f} - Accuracy: {acc:1.4f} - Jaccard: {jaccard:1.4f} - Recall: {recall:1.4f}")
#
#
#     fps = 