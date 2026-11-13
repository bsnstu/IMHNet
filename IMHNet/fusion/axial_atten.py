# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:17:13 2021

@author: angelou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from sklearn.metrics import confusion_matrix

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output
class AA_kernel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1,stride=1,padding=0)
        #self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3),stride = 1, padding=1)

        #self.layer4 = nn.Conv2d(512, 512, 3, padding=1)
        #self.bn4 = nn.BatchNorm2d(512)
        #self.relu= nn.PReLU()
        #self.pool = nn.MaxPool2d(2, 2)
        
        #self.Hattn = self_attn(out_channel, mode='h')
        
        #self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, cfp, decoder):
        cfp = self.conv0(cfp)
       
        #x=self.layer4(cfp)
        #x=self.bn4(x)
        #x=self.relu(x)
        #x=self.pool(x)
        decoder = -1 * (torch.sigmoid(decoder)) + 1
        attention = cfp.mul(decoder)
        #Hx = self.Hattn(x)
        #Wx = self.Wattn(Hx)
        return attention

class AA_edge_kernel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AA_edge_kernel, self).__init__()
        #self.conv0 = Conv(in_channel, out_channel, kSize=1,stride=1,padding=0)
        self.conv0=nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1,padding=0)

    def forward(self, cfp, decoder):
        cfp = self.conv0(cfp)
        decoder = -1 * (torch.sigmoid(decoder)) + 1
        attention = cfp.mul(decoder)
        #Hx = self.Hattn(x)
        #Wx = self.Wattn(Hx)
        return attention

'''
class AA_edge_kernel1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AA_edge_kernel1, self).__init__()
        #self.conv0 = Conv(in_channel, out_channel, kSize=1,stride=1,padding=0)
        self.conv0=nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1,padding=0)

    def forward(self, cfp, decoder):
        cfp1=cfp
        decoder1=decoder
        part1 = -1 * (torch.sigmoid(decoder1)) + 1
        part2 = -1 * (torch.sigmoid(cfp1)) + 1
        attention_edge = cfp.mul(part1)
        attention_area = decoder.mul(part2)
        #Hx = self.Hattn(x)
        #Wx = self.Wattn(Hx)
        return attention_edge,attention_area
'''