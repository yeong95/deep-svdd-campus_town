from collections import OrderedDict

import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

import torchvision
from torchvision.transforms import transforms

import sys
sys.path.append("..")
from base.base_net import BaseNet


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p, bias=False)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)
    

class Encoder(BaseNet):
    """
    Encoder class, mainly consisting of three residual blocks.
    """
    
    def __init__(self, trial):
        super(Encoder, self).__init__()
        self.rep_dim = trial.suggest_int("rep_dim", 4, 128) 

        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False) 
        self.BN = nn.BatchNorm2d(16)

        self.n_layers = trial.suggest_int("n_layers", 3, 6)
        logger = logging.getLogger()
        logger.info('Resblock layers: %d' % self.n_layers)
        in_channels = 16

        self.channels_list = []
        self.stride_list = []
        for i in range(self.n_layers):
            if i == (self.n_layers-1):
                out_channels = 2
            else:
                out_channels = trial.suggest_categorical("out_channels_{}".format(i+1), [16,32,48])
            stride = trial.suggest_categorical("stride_{}" .format(i+1), [1,2])
            var_name = 'self.rb' + str(i+1)
            globals()[var_name] = ResBlock(in_channels, out_channels, 3, stride, 1, 'encode')
            
            self.channels_list.append(in_channels)
            self.channels_list.append(out_channels)
            self.stride_list.append(stride)
            in_channels = out_channels 
        
        

        # self.rb1 = ResBlock(16, 16, 3, 2, 1, 'encode')
        # self.rb2 = ResBlock(16, 32, 3, 1, 1, 'encode')
        # self.rb3 = ResBlock(32, 32, 3, 2, 1, 'encode') 
        # self.rb4 = ResBlock(32, 48, 3, 1, 1, 'encode') 
        # self.rb5 = ResBlock(48, 48, 3, 2, 1, 'encode') 
        # self.rb6 = ResBlock(48, 2, 3, 2, 1, 'encode') 

        # self.fc1 = nn.Linear(, self.rep_dim, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1 = globals()['self.rb1'].cuda()(init_conv)

        for i in range(1, self.n_layers):
            var_name = 'rb' + str(i+1)
            self_rb = globals()['self.rb' + str(i+1)].cuda()
            rb_before = locals()['rb' + str(i)].cuda()
            locals()[var_name] = self_rb(rb_before).cuda()

        rb_last = locals()['rb'+str(i+1)]
        x = rb_last.view(rb_last.size(0),-1)
        x = nn.Linear(x.size()[1], self.rep_dim, bias=False).cuda()(x)
        # x = self.fc1(x)

        # rb2 = self.rb2(rb1)
        # rb3 = self.rb3(rb2)
        # rb4 = self.rb4(rb3)
        # rb5 = self.rb5(rb4)
        # rb6 = self.rb6(rb5)
        # x = rb6.view(rb6.size(0),-1)
        # x = self.fc1(x)
        return x

class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self, n_layers, channel_list, rep_dim, stride_list):
        super(Decoder, self).__init__()
        self.rep_dim = rep_dim
        self.channel_list = channel_list
        self.n_layers = n_layers
        self.stride_list = stride_list

        self.fc1 = nn.Linear(self.rep_dim, 2 * 40 * 40, bias=False)

        for i in range(self.n_layers):
            var_name = 'self.rb'+str(i+1)
            in_channels = self.channel_list.pop()
            out_channels = self.channel_list.pop()
            stride = self.stride_list.pop()

            globals()[var_name] = ResBlock(in_channels, out_channels, 3, stride, 1, 'decode')
        
        
        # self.rb1 = ResBlock(2, 48, 2, 2, 0, 'decode') 
        # self.rb2 = ResBlock(48, 48, 2, 2, 0, 'decode') 
        # self.rb3 = ResBlock(48, 32, 3, 1, 1, 'decode') 
        # self.rb4 = ResBlock(32, 32, 2, 2, 0, 'decode') 
        # self.rb5 = ResBlock(32, 16, 3, 1, 1, 'decode') 
        # self.rb6 = ResBlock(16, 16, 2, 2, 0, 'decode') 
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, inputs):

        fc1 = self.fc1(inputs)
        fc1 = fc1.view(fc1.size(0), 2, 40, 40)
        rb1 = globals()['self.rb1'].cuda()(fc1)        
        # rb1 = self.rb1(fc1)
  

        for i in range(1, self.n_layers):
            var_name = 'rb' + str(i+1)
            self_rb = globals()['self.rb'+str(i+1)].cuda()
            rb_before = locals()['rb' + str(i)].cdua()
            locals()[var_name] = self_rb(rb_before).cuda()

        # rb2 = self.rb2(rb1)
        # rb3 = self.rb3(rb2)
        # rb4 = self.rb4(rb3)
        # rb5 = self.rb5(rb4)
        # rb6 = self.rb6(rb5)

        rb_lst = locals()['rb'+str(i+1)]
        out_conv = self.out_conv(rb_lst)
        output = self.tanh(out_conv)
        import pdb;pdb.set_trace()
        print("autoencoder output shape : {}" .format(output.shape))
        return output
    
class Autoencoder(BaseNet):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self, trial):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(trial)
        self.decoder = Decoder(self.encoder.n_layers, self.encoder.channels_list, self.encoder.rep_dim, self.encoder.stride_list)
    
    # @property
    # def num_params(self):
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     num_p = sum([np.prod(p.size()) for p in model_parameters])
    #     return num_p
    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    tmp = torch.randn(4,3,640,640)
    net = Encoder()
    import pdb;pdb_set_trace()