
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]).cuda())   
        # self.iou = nn.Loss

    def forward(self, inputs, targets, smooth=0):

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        BCE = self.bce(inputs, targets)

        #flatten label and prediction tensors
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        # Dice_BCE = BCE #+ dice_loss

        return 5 * BCE + dice_loss
# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1e-5):
#
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         # inputs = torch.sigmoid(inputs)
#
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean',)
#         Dice_BCE = 5 * BCE + dice_loss
#
#         return Dice_BCE


def criterion(pred, traget, mod='BCELoss'):
    # pred.size = [b, 1]
    # traget.size = [b, 1]

    # nn.L1Loss  same dim
    # nn.SmoothL1Loss
    # nn.MSELoss

    # nn.BCELoss
    # + Sigmoid
    if mod == 'BCELoss':
        criterion = nn.BCELoss()
        loss = criterion(torch.sigmoid(pred), traget)

    if mod == 'Dice+BCE':
        criterion = DiceBCELoss()
        loss = criterion(torch.sigmoid(pred), traget)

    if mod == 'BCELoss':
        criterion = nn.BCELoss()
        loss = criterion(torch.sigmoid(pred), traget)

    if mod == 'DiceLoss':
        criterion = DiceLoss()
        loss = criterion(pred, traget)


    return loss
