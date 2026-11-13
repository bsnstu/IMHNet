# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .import vit_seg_configs as configs
from .vit_seg_modeling_Unet import UNet


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        
        
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # (B,196,12,64)
        x = x.view(*new_x_shape)  
        return x.permute(0, 2, 1, 3)  #(B,12,196,64)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) 
        mixed_key_layer = self.key(hidden_states) 
        mixed_value_layer = self.value(hidden_states) 
        
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (B,12,196,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (B,12,196,64)matmul(B,12,64,196)=(B,12,196,196)
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer) # (B,12,196,64)  #(B,12,196,196)matmul(B,12,196,64)=(B,12,196,64)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # (B,196,768)
        
        attention_output = self.out(context_layer) #(B,196,768)
        attention_output = self.proj_dropout(attention_output) #(B,196,768)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:

            grid_size = config.patches["grid"] #(14,14)
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1]) #(1,1)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)#16*16
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  #14*14
            self.hybrid = True

        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = True


        if self.hybrid:
            #self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            self.hybrid_model=UNet(1,3)
            #in_channels= 1024
            #in_channels = self.hybrid_model.width * 16

        
        self.patch_embeddings = Conv2d(in_channels=in_channels, #1024
                                       out_channels=config.hidden_size, #768
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x4,features,x_edge_final = self.hybrid_model(x)  # (B,C,H/2,W/2)(B,C,H/4,W/4)(B,C,H/8,W/8)

        else:
            features = None
        
        x=x[:,0,:,:].unsqueeze(1)
        x=x.repeat(1,3,1,1)
        x = self.patch_embeddings(x) 
        x = x.flatten(2) # (B, hidden, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features, x4, x_edge_final


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x # (B, n_patchs, hidden):(B, 196, 768)
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[(ROOT+'/'+ATTENTION_Q+"/kernel")]).view(self.hidden_size, self.hidden_size).t() 
            key_weight = np2th(weights[(ROOT+'/'+ATTENTION_K+"/kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[(ROOT+'/'+ ATTENTION_V+"/kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[(ROOT+'/'+ATTENTION_OUT+"/kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[(ROOT+'/'+ATTENTION_Q+"/bias")]).view(-1)
            key_bias = np2th(weights[(ROOT+'/'+ ATTENTION_K+"/bias")]).view(-1)
            value_bias = np2th(weights[(ROOT+'/'+ATTENTION_V+"/bias")]).view(-1)
            out_bias = np2th(weights[(ROOT+'/'+ATTENTION_OUT+"/bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[(ROOT+'/'+FC_0+"/kernel")]).t()
            mlp_weight_1 = np2th(weights[(ROOT+'/'+FC_1+"/kernel")]).t()
            mlp_bias_0 = np2th(weights[(ROOT+'/'+FC_0+"/bias")]).t()
            mlp_bias_1 = np2th(weights[(ROOT+'/'+FC_1+"/bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[(ROOT+'/'+ATTENTION_NORM+"/scale")]))
            self.attention_norm.bias.copy_(np2th(weights[(ROOT+'/'+ ATTENTION_NORM+"/bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[(ROOT+'/'+MLP_NORM+"/scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[(ROOT+'/'+MLP_NORM+"/bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states): #hidden_states:# (B, n_patches, 768):(B, 196, 768)
        attn_weights = []
        
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights # encoded：(B,196,768); attn_weights:None


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size) 
        self.encoder = Encoder(config, vis)
    #x->(B 3 H W)
    def forward(self, input_ids):
        embedding_output, features, Resout_4, x_edge_final = self.embeddings(input_ids)  # embedding_out:(B, 196, 768)  features(B,512,H/8,W/8);(B,256,H/4,W/4);(B,64,H/2,W/2)
        encoded, attn_weights = self.encoder(embedding_output)  
        return encoded, attn_weights, features,Resout_4,x_edge_final


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):       # in_channels=[512,256,128,64] out_channels=[256,128,64,16] skip_channels=[512, 256, 64, 0]
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,skip=None): # x:(B,512,H/16,W/16)  skip:(B,512,H/8,W/8);(B,256,H/4,W/4);(B,64,H/2,W/2)
        
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x,skip], dim=1)
        
        x = self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        return x
class DecoderBlock3(nn.Module):       # in_channels=[512,256,128,64] out_channels=[256,128,64,16] skip_channels=[512, 256, 64, 0]
    def __init__(
            self,
            one_channels,
            two_channels,
            three_channels,
            four_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(one_channels+two_channels+three_channels, two_channels+three_channels, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(two_channels+three_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(two_channels+three_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,skip=None,y=None,z=None): # x:(B,512,H/16,W/16)  skip:(B,512,H/8,W/8);(B,256,H/4,W/4);(B,64,H/2,W/2)
        
        x = self.up(x)
        y=y+z
        if skip is not None:
            x = torch.cat([x,skip,y], dim=1)
        
        x = self.conv1(x)
        #x=self.bn1(x)
        #x=self.relu(x)
        return x


class DecoderBlock2(nn.Module):       # in_channels=[512,256,128,64] out_channels=[256,128,64,16] skip_channels=[512, 256, 64, 0]
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.neck=nn.Sequential(
            nn.Conv2d(1536,1536,kernel_size=1, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536, 1536, kernel_size=1, stride=1,padding=0)
        )
        self.convRelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )
    def forward(self, x, y): # x(CNN):1024 y(Transformer):512
        
        z=torch.cat((x,y),dim=1)
        
        z_spatial,_=torch.max(z,dim=1,keepdim=True)
        z_spatial=torch.sigmoid(z_spatial)
        z_atten=z*z_spatial
        z_atten=self.neck(z_atten)
        z_add=z+z_atten
        x_y=self.convRelu(z_add)
        return x_y

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


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class down_conv_EGFP(nn.Module):
    def __init__(self, in_ch, out_ch, skip):
        super(down_conv_EGFP, self).__init__()
        self.up = nn.Sequential(
            #nn.MaxPool2d(skip, skip, ceil_mode=True),
            nn.Upsample(scale_factor=1/skip, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.up(x)
        return x


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

class ESAB(nn.Module):
    def __init__(self, in_channels):
        super(ESAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.ban = nn.BatchNorm2d(1)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, 1)

    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x)
        y = F.relu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        #y = F.relu(self.ban(y))
        return y
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,  #768
            head_channels,   #512
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels #[256,128,64,16]
        in_channels = [512,256,128,64]#inchannels=[512,256,128,64]
        out_channels = [256,128,64,16] #out_channels=[256,128,64,16]

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels #[512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0 #[512, 256, 64, 0]

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.blocks1=DecoderBlock3(512,512,256,256,256) #three:256
        self.blocks2=DecoderBlock3(256,256,128,128,128) #128
        self.blocks3 = DecoderBlock3(128,64,64,64,64)  # 64
        self.blockscnn=DecoderBlock(1024,256,512)
       # self.block1=DecoderBlock1(1024,256)
        self.block2=DecoderBlock2(1536,512)
        self.down_edge_8=down_conv_EGFP(64,512,8)
        self.down_edge_4=down_conv_EGFP(16,128,4)
        self.down_edge_2=down_conv_EGFP(16,64,2)
        
        
        self.up_x_fusion=up_conv_skip(384,64,2)
        self.up_x_fusion1 = up_conv_skip(256, 64, 2)  
        self.conv_edge1 = conv_block(128, 64)  
        
        self.up_2_edge=up_conv(64,16)
        self.up_input_edge=nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        #edge||area 32+32->32
        self.ea_to_a=nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.con_2=nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
    def forward(self, hidden_states, features=None, Resout_4=None, x_edge_final=None):  
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch)) #h 14 w 14
        x = hidden_states.permute(0, 2, 1)
        
        x = x.contiguous().view(B, hidden, h, w)# (B,768,14,14)
        
        x_trans = self.conv_more(x)  
        
        x4_cnn=Resout_4 #4 1024 14 14
        
      #  x_fusion_up = self.up_x_fusion1(features[1])
       # x_edge = self.conv_edge1(torch.cat((features[2], x_fusion_up), dim=1))  # 64+64->64 H/2 W/2

        
        
        
        x_4to3_cnn=self.blockscnn(x4_cnn,features[0])#256
        x_3to2_cnn=self.blocks[1](x_4to3_cnn,features[1])
        x_2to1_cnn = self.blocks[2](x_3to2_cnn, features[2])

        x_4to3_trans=self.blocks[0](x_trans,features[0])#256
        x_3to2_trans = self.blocks[1](x_4to3_trans,features[1])
        x_2to1_trans = self.blocks[2](x_3to2_trans, features[2])


        x_Trans_Cnn = self.block2(x4_cnn, x_trans)  
        
        x_4to3_fusion=self.blocks1(x_Trans_Cnn,features[0],x_4to3_trans,x_4to3_cnn)
        x_3to2_fusion = self.blocks2(x_4to3_fusion,features[1],x_3to2_trans,x_3to2_cnn)
        x_2to1_fusion = self.blocks3(x_3to2_fusion,features[2],x_2to1_trans,x_2to1_cnn)
        
        x_1_contact = self.con_2(torch.cat([x_2to1_fusion, x_2to1_cnn, x_2to1_trans], dim=1))  # 64
        
        x_1toinput = self.blocks[3](x_1_contact, skip=None)  # 16 up 2->H W

        x_total = x_edge_final + x_1toinput  # 16
        x_edge_input = self.up_input_edge(x_edge_final)
        x_final_area = self.ea_to_a(x_total)

        return x_final_area, x_edge_input



class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1], #16
            out_channels=config['n_classes'], #1
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        
        x, attn_weights, features ,Resout_4, x_edge_final = self.transformer(x)  # (B, n_patch, hidden)
        
        x_area,x_edge1= self.decoder(x, features, Resout_4,x_edge_final) # (B,16,H,W):(B,16,224,224)
        
        #logits = self.segmentation_head(x)
        return x_area, x_edge1

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
            '''
            if self.transformer.embeddings.hybrid: 
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
            '''
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


