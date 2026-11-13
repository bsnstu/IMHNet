from Configure import *
import os
import time
# from glob import glob
# from operator import add
import torch
from tqdm import tqdm
# from dataloader import load_data
from dataloader_five_fold import load_data
from model import network, network_pp
# from runner.losses import DiceLoss, DiceBCELoss
from runner.utils import seeding, create_dir, epoch_time
# from test import calculate_metrics, mask_parse
from runner.losses import criterion, DiceBCELoss
from runner.metrics import calculate_metrics
from runner.visualization import visualization_segmentation as vis
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Solver(object):
    def __init__(self, cfg, _train_loader, _valid_loader):
        self.cfg = cfg
        self.model = network().cuda()
        # self.model = network_pp(2).cuda()  ## unet_pp
        self.train_loader = _train_loader
        self.valid_loader = _valid_loader

    def train_one_epoch(self, data_loader, model, optimizer, scheduler, epoch):
        model.train()
        # init metrics

        epoch_loss = 0.0
        metrics_dict = {'MET': []}
        dbloss = DiceBCELoss()
        loop = tqdm(data_loader, total=len(data_loader), desc='[CNN_BASE-EPOCH %s]' % epoch, leave=True)
        for step, samples in enumerate(loop):
            image, mask = samples
            pred = model(image)
            loss = criterion(pred, mask, mod='DiceLoss')
            # loss = dbloss(torch.sigmoid(pred), mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            metrics = calculate_metrics(torch.sigmoid(pred.data).cpu().numpy(), mask.data.cpu().numpy())
            metrics_dict['MET'].append(metrics['MET']['value'])
            loop.set_postfix()
            vis(image[0, 0, ...], torch.sigmoid(pred.data)[0, 0, ...], mask[0, 0, ...], '{}_{}'.format(epoch, step),
                phase='train')

        epoch_loss = epoch_loss/len(loop)
        
        met = np.vstack(metrics_dict['MET']).mean(axis=0)

        f1, acc = met[2], met[3]

        return epoch_loss, f1, acc,  met

    def evaluate(self, data_loader, model, epoch):
        epoch_loss = 0.0
        metrics_dict = {'MET': []}
        dbloss = DiceBCELoss()
        model.eval()
        with torch.no_grad():
            for step, samples in enumerate(data_loader):
                image, mask = samples

                pred = model(image)
                loss = criterion(pred, mask, mod='DiceLoss')
                # loss = dbloss(torch.sigmoid(pred), mask)
                epoch_loss += loss.item()

                metrics = calculate_metrics(torch.sigmoid(pred.data).cpu().numpy(), mask.data.cpu().numpy())

                metrics_dict['MET'].append(metrics['MET']['value'])
                vis(image[0, 0, ...], torch.sigmoid(pred.data)[0, 0, ...], mask[0, 0, ...], '{}_{}'.format(epoch, step),
                    phase='valid')


            met = np.vstack(metrics_dict['MET']).mean(axis=0)

            f1, acc = met[2], met[3]

            return epoch_loss/len(data_loader), f1, acc,  met

    def running(self):

        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.base_lr, weight_decay=5e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6, patience=5, verbose=True)

        """ Training the model """
        best_val_f1 = 0.
        
        log = {'train': [], 'val': []}

        for epoch in range(1, CFG.epochs):
            start_time = time.time()

            # TRAIN
            train_loss, train_f1, train_acc, train_met = self.train_one_epoch(self.train_loader, model, optimizer, scheduler, epoch)
            if(epoch%10==0)or (epoch==99):
                valid_loss, val_f1, val_acc, val_met = self.evaluate(self.valid_loader, model, epoch)
                scheduler.step(valid_loss)
                
                log['train'].append(np.hstack([[epoch], train_met]))
                log['val'].append(np.hstack([[epoch], val_met]))

                """ Saving the model """
                if val_f1 >=best_val_f1:
                    data_str = f"\nValid f1 improved from {best_val_f1:2.4f} to {val_f1:2.4f}. Saving checkpoint: {CFG.checkpoint_path}"
                    print(data_str)
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), '%s/Myo_new2.pth' % (CFG.checkpoint_path))

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                data_str = f'\nEpoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
                data_str += f'\tTrain Loss: {train_loss:.3f} \tTrain F1: {train_f1:.3f} \tTrain ACC: {train_acc:.3f}\n'
                data_str += f'\tVALID Loss: {valid_loss:.3f} \tVALID F1: {val_f1:.3f} \tVALID ACC: {val_acc:.3f}\n'
                print(data_str)

        log_train = pd.DataFrame(np.vstack(log['train']),
                                 columns=['EPOCH', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Jaccard'])
        log_val = pd.DataFrame(np.vstack(log['val']),
                               columns=['EPOCH',  'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Jaccard'])

        log_train.to_csv(os.path.join(CFG.SAVE_ROOT, 'log_train.csv'),
                         header=['EPOCH',  'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Jaccard'],
                         index=False)
        log_val.to_csv(os.path.join(CFG.SAVE_ROOT, 'log_val.csv'),
                       header=['EPOCH',  'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Jaccard'],
                       index=False)


if __name__ == "__main__":

    train_loader = load_data(phase='train')
    valid_loader = load_data(phase='val')
    solver = Solver(CFG, train_loader, valid_loader)
    solver.running()
