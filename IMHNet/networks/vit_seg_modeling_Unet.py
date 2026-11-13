import torch
from torch import nn
import numpy as np
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
class SWMBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        #self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
       # self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv1(x)
       # out = self.bn1(out)
       # out = self.relu(out)

        out = self.conv2(out)
       # out = self.bn2(out)
        #out = self.relu(out)
        #out=torch.sigmoid(out)
        return out

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input
def run_sobel1(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return g
class ESAB(nn.Module):
    def __init__(self, in_channels):
        super(ESAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(1)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, 1)

    def forward(self, x):
        y = run_sobel1(self.sobel_x1, self.sobel_y1, x)
       # y = F.relu(self.bn(y))
        y = self.conv1(y)
       # y = x + y
        y = self.conv2(y)
        #y = F.relu(self.ban(y))
        return y
class ESAB1(nn.Module):
    def __init__(self, in_channels):
        super(ESAB1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(1)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, 1)

    def forward(self, x):
        y = run_sobel1(self.sobel_x1, self.sobel_y1, x)
        #print(torch.max(y))
       # y = F.relu(self.bn(y))
        y = self.conv1(y)
       # y = x + y
        y = self.conv2(y)
        #y = F.relu(self.ban(y))
        return y
class up_conv_skip(nn.Module):
    def __init__(self, in_ch, out_ch, skip=2):
        super(up_conv_skip, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=skip,mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  
        self.conv0= nn.Conv2d(1,1,kernel_size=3, stride=1, padding=1)
        self.conv0_0 = VGGBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv_SWM_0=SWMBlock(2,1,1)

        self.conv_SWM_1=SWMBlock(1, nb_filter[0], nb_filter[0])
        self.conv_SWM_2=SWMBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv_SWM_3=SWMBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv_SWM_4=SWMBlock(nb_filter[2], nb_filter[3], nb_filter[3])



       # self.sobel=ESAB(32)
        self.input_sobel=ESAB1(1)
        self.edge_sobel=ESAB(1)
        self.conv=nn.Conv2d(1,32,1,1,0)
        #self.sobel_conv=nn.Conv2d(33,32,1,1,0)

        self.up_x_fusion1 = up_conv_skip(64, 32, 2)  
        self.conv_edge1 = conv_block(64, 32)  
        
        self.up_2_edge = up_conv(32, 32)  
        self.up_input_edge = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

        self.conv_edge2=conv_block(32, 16)  
        #self.conv_edge3=conv_block(32,32)
    def forward(self, input):
        features=[]

        Train_f=input[:,2,:,:].unsqueeze(1)
        weight_f=input[:,1,:,:].unsqueeze(1)
        input_f=input[:,0,:,:].unsqueeze(1)

        
        #weight_0=self.convf_0(weight_0)
        x_inputsobel = self.input_sobel(input_f) #1
        weight_0=self.conv_SWM_0(torch.cat((weight_f,Train_f),dim=1))#2->1
        
        weight_mul=weight_0*x_inputsobel
        weight_0=weight_mul+weight_0
        weight_0_f=torch.sigmoid(weight_0) #1
        
       # x_inputsobel=self.input_sobel(input_f)
        f_input=weight_0_f*input_f
        #print(f_input)
        x0_0 = self.conv0_0(f_input) #1->32 H W

        weight_1 = self.conv_SWM_1(self.pool(weight_0))  # 32
        weight_1_f=torch.sigmoid(weight_1+self.pool(x0_0))
        x1_0_w_f=weight_1_f*self.pool(x0_0)
        x1_0 = self.conv1_0(x1_0_w_f) # 32->64 H/2 W/2

        features.append(x1_0)

        weight_2=self.conv_SWM_2(self.pool(weight_1))#64
        weight_2_f=torch.sigmoid(weight_2+self.pool(x1_0))
        x2_0_w_f=weight_2_f*self.pool(x1_0)
        x2_0 = self.conv2_0(x2_0_w_f)  # 64->256 H/4 W/4
        features.append(x2_0)

        weight_3 = self.conv_SWM_3(self.pool(weight_2))  #
        weight_3_f = torch.sigmoid(weight_3+self.pool(x2_0))
        x3_0_w_f = weight_3_f * self.pool(x2_0)
        x3_0 = self.conv3_0(x3_0_w_f)  # 256->512 H/8 W/8
        features.append(x3_0)

        weight_4=self.conv_SWM_4(self.pool(weight_3))
        weight_4_f=torch.sigmoid(weight_4+self.pool(x3_0))
        x4_0_w_f=weight_4_f*self.pool(x3_0)
        x4_0=self.conv4_0(x4_0_w_f)
        
        x_fusion_up = self.up_x_fusion1(x1_0)
        x_edge = self.conv_edge1(torch.cat((x0_0, x_fusion_up), dim=1))  # 32+32->64->32H/2 W/2
        
        #x_edge=self.sobel(x_edge)
        
        #input_sobel=self.edge_sobel(input_f)#32
        x_edge_sobel=torch.sigmoid(weight_mul)*x_edge+x_edge
        #x_edge=x_edge+self.conv_edge3(x_edge_sobel)
        
       # x_edge_ = self.up_2_edge(x_edge)  # 64->32 H W
        
        x_edge=self.conv_edge2(x_edge_sobel)#64->16
        #x_edge_final1 = self.up_input_edge(x_edge)  # 16->1
        #x4_0 = self.conv4_0(self.pool(x3_0)) #512->1024 H/16 W/16
        #weight_4 = self.conv_ASW_4(self.pool(weight_3))  # 64
        #x4_0_w_f=weight_4*x4_0
        return x4_0, features[::-1],x_edge
