"""
MLSCAlib
Copyright (C) 2025 CSEM 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#
# brief     : Pytorch-based NN architectures to attack AES
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

#
import math
import time

from error import Error
import torch
import functorch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEFAULT_STANDARD_DEVIATION = 0.1

def _log_softmax(dim,training=True):
    if(dim == "no softmax" or dim=="no"):
        return lambda x : x
    elif(dim == "selu"):
        return F.selu
    if(dim >= 3):
        dim = dim % 2
        if(not training):
            dim = 1 - dim
    if(dim == 2):
        return lambda x : (torch.nn.LogSoftmax(0)(x) + torch.nn.LogSoftmax(1)(x))
    elif(dim == 1):
        return lambda x : torch.nn.LogSoftmax(dim)(x)
    elif(dim == 0):
        return lambda x : torch.nn.LogSoftmax(dim)(x)
    else:
        raise Error("Invalid dimension")
    

class _PreActBottleneck(nn.Module):
    """PreActBottleneck: a ResNet subpart containing shortcuts for the gradient."""
    expansion = 4
    def __init__(self, in_planes, planes,kernel_size=3, stride=1,conv_shortcut=False):
        super(_PreActBottleneck, self).__init__()
        self.in_planes=in_planes
        self.bn1 = nn.BatchNorm1d(in_planes)  #1 was in_planes
        if(conv_shortcut):
            self.shortcut = nn.Conv1d(in_planes,self.expansion * planes,kernel_size =1, stride=stride)  
            #self.conv1 = nn.Conv1d(in_channels=self.expansion * planes, out_channels=planes, kernel_size=1, bias=False) 
        elif(stride > 1): 
            self.shortcut = nn.Sequential(\
                nn.MaxPool1d(1,stride=stride)  #do NOT use AVGPOOLING here
                #nn.Conv1d(in_planes, self.expansion*planes,
                 #         kernel_size
                 # =1, stride=stride, bias=False)
            )
        self.conv1 = nn.Conv1d(in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)  #bias = True breaks everything
    def forward(self, x):
        out = (self.bn1(x))
        out=F.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    """ResNet architecture.
    
    This architecture has been proposed by :footcite:t:`[211]burszteindc27` and is
    available here :footcite:t:`[218]bursztein2019scaaml`.
    
    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    
    
    """
    def __init__(self, num_classes,ns,dk,noise_std,dim = 0,num_blocks = [3,4,6,3]):
        super(ResNet, self).__init__()
        self.in_planes = 256 # or 64 ? 
        self.dim = dim
        self.dk = dk
        self.noise = noise_std
        #MaxPool1d
        #self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1,
         #                      padding=1, bias=False)
        self.pool=nn.MaxPool1d(6) #from video, should be 6

        #conv(16) / batch / conv (filters)/ batch
        self.layer1 = self._make_layer( 64, num_blocks[0],first=True)#, strides=1)
        
        self.layer2 = self._make_layer( 128, num_blocks[1])#, stride=2)
        self.layer3 = self._make_layer( 256, num_blocks[2])#, stride=2)
        self.layer4 = self._make_layer( 512, num_blocks[3])#, stride=2)
        self.drop = nn.Dropout()   
        self.linear0= nn.Linear(512*_PreActBottleneck.expansion ,num_classes)
        self.batch = nn.BatchNorm1d(num_classes)

        self.linear = nn.Linear(num_classes, num_classes)
        self.out = _log_softmax(dim = self.dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def _make_layer(self,filters_in_planes,num_blocks,kernel_size=3,strides=2,activation="relu",first=False):
    # block, planes, num_blocks, stride):
        #strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        if(first):
            num_1 = 1
        else:
            #self.in_planes *= 2
            num_1 = self.in_planes
        layers.append(_PreActBottleneck(num_1, filters_in_planes,kernel_size = kernel_size,stride=strides,conv_shortcut=True))
        if(not first):
            self.in_planes *= 2
        for i in range(2,num_blocks):
            layers.append(_PreActBottleneck(self.in_planes,filters_in_planes,kernel_size=kernel_size))

        layers.append(_PreActBottleneck(self.in_planes,filters_in_planes,stride=strides))
        #for stride in strides:
         #   layers.append(PreActBottleneck(self.in_planes, planes, stride))
          #  self.in_planes = planes * PreActBottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],1,x_dk.size()[2]))
        x = self.pool(x)
        out_tmp=x
        out_tmp = self.layer1(out_tmp)
        out_tmp = self.layer2(out_tmp)
        out_tmp = self.layer3(out_tmp)
        out_tmp = self.layer4(out_tmp)
        if(out_tmp.shape[2]>1):
            out_tmp = F.avg_pool1d(out_tmp, 2)
        out_tmp = out_tmp.view(out_tmp.size(0), -1)
        #
        out_tmp=self.drop(out_tmp)
        if(self.dk):
           # print("shape x is ",x.shape,x_dk.shape)
            x=torch.cat((x,x_dk),-1)
        out_tmp=self.linear0(out_tmp)
        out_tmp=self.batch(out_tmp)
        out_tmp=F.relu(out_tmp)
        out_tmp = self.linear(out_tmp)
        return self.out(out_tmp)

class VGG16(nn.Module):
    """A VGG model, as taken from :footcite:t:`[22]Kim_Picek_Heuser_Bhasin_Hanjalic_2019`.
    
    Recommended to use a L2 reg of value 10^-7
    
    

    """
    def __init__(self,num_classes,ns,dk=False,noise_std=None,dim=0):
        super(VGG16, self).__init__()
        self.dk=dk
        self.dim=dim
        self.noise_std = noise_std
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1) #original input channels : 3, conv2d
        self.bn1 = nn.BatchNorm1d(8)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, padding=1)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(in_channels=160, out_channels=128, kernel_size=3, padding=1)

        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.drop = nn.Dropout()
        c = -3 + 2 + 1
        input_size = ((( ns +c ) // 2) + 7*c) * (256 + 128)
        if(self.dk):
            input_size += 256
        self.regularized_function_1 = nn.Linear(input_size, 512) #originaly: (25088, 4096)
        self.regularized_function_2 = nn.Linear(512, 512)  #not used
        self.fc3 = nn.Linear(512, num_classes)
        # adding output layer
        self.out = _log_softmax(dim = self.dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def _add_noise(self,x):
        if(self.noise_std is not None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            return x + gaussian_noise  
        else:
            return x
    def forward(self, x):
        #input x should be std
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))
        x = (self.conv1(x))
        x = F.selu(x)
        x = self.bn1(x)
        x = self._add_noise(x)
        x = self.maxpool(x)
        short = x
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = self.bn2(x)
        short2 = x
        x = torch.cat((x,short),1)
        x = self._add_noise(x)
        x = F.selu(self.conv4(x))
        #x = self.maxpool(x)
        x = F.selu(self.conv5(x))
        x = self.bn3(x)
        short3 = x
        x = torch.cat((x,short2),1)
        x = self._add_noise(x)
        x = F.selu(self.conv6(x))
        x = F.selu(self.conv7(x))
        x = self.bn4(x)
        x = self._add_noise(x)
        #x = self.maxpool(x)
        x = F.selu(self.conv8(x))
        x = torch.cat((x,short3),1)
        # print(x.shape)

        # x = self.drop(x)
       # print("Before shape",x.shape)
        x = x.reshape(x.shape[0], -1)
        #print("After shape ",x.shape)
        if(self.dk):
            x=torch.cat((x,x_dk),-1)

        x = F.selu(self.regularized_function_1(x))
        # x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.selu(self.regularized_function_2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return self.out(x)  #added a logsoftmax layer

class VGGNoise(nn.Module):
    """A VGG model, as taken from :footcite:t:`[22]Kim_Picek_Heuser_Bhasin_Hanjalic_2019`.
    
    
    Authors recommended to use a L2 reg of value 10^-7.
    
    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    
    
    """
    def __init__(self,num_classes,ns,dk=False,noise_std=None,dim=0):
        super(VGGNoise, self).__init__()
        self.dk=dk
        self.dim=dim
        self.noise_std = noise_std
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1) #original input channels : 3, conv2d
        self.bn1 = nn.BatchNorm1d(8)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.drop = nn.Dropout()
        c = -3 + 2 + 1
        input_size = ((( ns +c ) // 2) + 7*c) * (256)
        if(self.dk):
            input_size += 256
        self.regularized_function_1 = nn.Linear(input_size, 512) #originaly: (25088, 4096)
        self.regularized_function_2 = nn.Linear(512, 512)  #not used
        self.fc3 = nn.Linear(512, num_classes)
        # adding output layer
        self.out = _log_softmax(dim = self.dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def _add_noise(self,x):
        if(self.noise_std is not None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            return x + gaussian_noise  
        else:
            return x
    def forward(self, x):
        #input x should be std
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))
        x = (self.conv1(x))
        x = F.selu(x)
        x = self.bn1(x)
        x = self._add_noise(x)
        x = self.maxpool(x)
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = self.bn2(x)
        x = self._add_noise(x)
        x = F.selu(self.conv4(x))
        #x = self.maxpool(x)
        x = F.selu(self.conv5(x))
        x = self.bn3(x)
        x = self._add_noise(x)
        x = F.selu(self.conv6(x))
        x = F.selu(self.conv7(x))
        x = self.bn4(x)
        x = self._add_noise(x)
        #x = self.maxpool(x)
        x = F.selu(self.conv8(x))

        # x = self.drop(x)
       # print("Before shape",x.shape)
        x = x.reshape(x.shape[0], -1)
        #print("After shape ",x.shape)
        if(self.dk):
            x=torch.cat((x,x_dk),-1)

        x = F.selu(self.regularized_function_1(x))
        # x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.selu(self.regularized_function_2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return self.out(x)  #added a logsoftmax layer
    
class MCNN(nn.Module):
    """MCNN: multi-scale convolutional neural networks (MCNN).
    
     Inspired from https://github.com/mitMathe/SCA-MCNN.
    
    Model inspired from :footcite:t:`[56]9419959` and :footcite:t:`[219]MCNNRepo`.
    Includes preprocessing steps inside the architecture. Should be working well on any kind of implementation.
    
    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    

    """
    def __init__(self,num_classes,ns,dk=False,noise_std=None,dim=0,third_processing = "PCA"):
        super(MCNN, self).__init__()
        self.dk=dk
        self.dim=dim
        self.ns=ns
        self.is_first_epoch=True
        self.third_processing = third_processing
        self.noise_std=noise_std
        self.conv_123_1 = nn.Conv1d(in_channels=1,out_channels=32,kernel_size=1,padding='same',stride=1)
        self.batch_norm_123_1 = nn.BatchNorm1d(32)
        self.pool_123_1 = nn.AvgPool1d(2,stride=2)
        self.conv_12_2 = nn.Conv1d(32,out_channels=64,kernel_size=50,padding='same',stride=1)
        self.conv_3_2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=1,padding='same',stride=1)
        self.batch_norm_123_2 = nn.BatchNorm1d(64)
        self.pool_12_2 = nn.AvgPool1d(kernel_size =50,stride=50)
        self.pool_3_2 = nn.AvgPool1d(kernel_size =1,stride=1)
        self.batch_norm_123_3 = nn.BatchNorm1d(64)
        self.batch_norm_all_1 = nn.BatchNorm1d(64)
        self.conv_all_1 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding='same',stride=1)
        self.batch_norm_all_2 = nn.BatchNorm1d(128)
        self.pool_all_1 = nn.AvgPool1d(2,2)
        self.regularized_function_1 = nn.Linear
        self.dense_2 = nn.Linear(20,10)  
        self.regularized_function_2 = nn.Linear(10,20)  # We use 10 neurons instead of 20 here
        self.dense_4 = nn.Linear(20,num_classes)
        self.out = _log_softmax(dim = self.dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self,x):
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))
        x_1 = MCNN._get_MA(x)
        x_2 = x
        if(self.third_processing == "PCA"):
            x_3 = MCNN._get_PCA(x)
        else:
            raise Error(self.third_processing, "not implemented.")
        x_1 = F.selu(self.conv_123_1(x_1))
        x_2 = F.selu(self.conv_123_1(x_2))
        x_3 = F.selu(self.conv_123_1(x_3))
        
        x_1=self.batch_norm_123_1(x_1)
        x_2=self.batch_norm_123_1(x_2)
        x_3=self.batch_norm_123_1(x_3)

        x_1=self.pool_123_1(x_1)
        x_2=self.pool_123_1(x_2)
        x_3=self.pool_123_1(x_3)

        x_1 = F.selu(self.conv_12_2(x_1))
        x_2 = F.selu(self.conv_12_2(x_2))
        x_3 = F.selu(self.conv_3_2(x_3))

        x_1=self.batch_norm_123_2(x_1)
        x_2=self.batch_norm_123_2(x_2)
        x_3=self.batch_norm_123_2(x_3)
        if(x_1.shape[2]>=50):
            x_1 = self.pool_12_2(x_1)
        if(x_2.shape[2]>=50):
            x_2 = self.pool_12_2(x_2)
        x_3 = self.pool_3_2(x_3)

        x_1=self.batch_norm_123_3(x_1)
        x_2=self.batch_norm_123_3(x_2)
        x_3=self.batch_norm_123_3(x_3)

        x_cat = torch.cat((x_1,x_2,x_3),2)
        x_cat= self.batch_norm_all_1(x_cat)
        x_cat=F.selu(self.conv_all_1(x_cat))
        x_cat = self.batch_norm_all_2(x_cat)

        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x_cat.size()).to(x.device)
            x_cat = x_cat + gaussian_noise

        x_cat = self.pool_all_1(x_cat)
        x_cat = x_cat.view(x_cat.size(0), -1)
        if(self.dk):
            x_cat=torch.cat((x_cat,x_dk),-1)

        num_samples_x_cat = x_cat.shape[1]
        next_func = self.regularized_function_1(num_samples_x_cat,20).to(x.device)
        x_cat = F.selu(next_func(x_cat))
        x_cat = F.selu(self.dense_2(x_cat))
        x_cat = F.selu(self.regularized_function_2(x_cat))
        x_cat = F.selu(self.dense_4(x_cat))

        x_cat = self.out(x_cat)

        return x_cat

    @staticmethod
    def _moving_average_sub(data_x, window_size):
        no = data_x.shape[0]
        len = data_x.shape[1]
        out_len = len - window_size + 1
        output = torch.zeros((no, out_len))
        for i in range(out_len):
            output[:,i]=torch.mean(data_x[:,i : i + window_size], axis=1)

        return output

    @staticmethod
    def _moving_average(data_x, window_base=100, step_size=1, no=1):
        window_base = min(window_base,data_x.shape[1] - 1)
        if no == 0:
            return (None, [])
        out = MCNN._moving_average_sub(data_x, window_base)
        data_len = [out.shape[1]]
        for i in range(1, no):
            window_size = window_base + step_size * i
            if window_size > data_x.shape[1]:
                continue
            new_series = MCNN._moving_average_sub(data_x, window_size)
            data_len.append(new_series.shape[1])
            out = torch.cat([out, new_series], axis=1)
        return (out, data_len)
    @staticmethod
    def _min_max(x):
        def minmax(v):
            v_min, v_max = torch.min(v), torch.max(v)
            new_min, new_max = 0., 1.
            v_p = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
            return v_p
        f = functorch.vmap(lambda v : minmax(v))
        return f(x)
    @staticmethod
    def _finish_process(x):
        mn = torch.mean(x, axis=1, keepdims=True)
        std = torch.std(x, axis=1, keepdims=True)
        x = (x - mn) / std                       #investigate :should it use the standardization ? 
        x=MCNN._min_max(x)
        x = x.reshape((x.shape[0],1,x.shape[1]))
        return x
    @staticmethod
    def _get_MA(x):
        x_re = x.reshape(x.shape[0],x.shape[2])
        x_ma,len_ma =  MCNN._moving_average(x_re)
        return MCNN._finish_process(x_ma).to(x.device)
    @staticmethod
    def _get_PCA(x):
        x_re = x.reshape(x.shape[0],x.shape[2])
        x_pca,_,_ = torch.pca_lowrank(x_re,q=20)
        return MCNN._finish_process(x_pca).to(x.device)

class NetAeshd(nn.Module):
    """ NetAeshd: neural network proposed in :footcite:t:`[3]Zaid_Bossuet_Habrard_Venelli_2019` to break the AES_HD dataset, an unprotected AES implementation.
    
    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    
    """

    def __init__(self, num_classes, ns,dk=False,noise_std=None,dim=0):
        super(NetAeshd, self).__init__()
        if(dk):
            ns+=256
        self.dk=dk
        self.dim=dim
        self.noise_std=noise_std
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1)
        self.batch1 = nn.BatchNorm1d(2)
        self.batch2 = nn.BatchNorm1d(1)
        self.pool = nn.AvgPool1d(2, 2)
        self.regularized_function_1 = nn.Linear(ns, 2)
        self.regularized_function_2 = nn.Linear(2, num_classes)
        self.out = _log_softmax(dim = self.dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x):
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))
        x = self.conv1(x)
        x = F.selu(x)
        x = self.batch1(x)
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise      
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if(self.dk):
            x=torch.cat((x,x_dk),-1)
        x = F.selu(self.regularized_function_1(x))
        x = self.regularized_function_2(x)
        x = _log_softmax(self.dim,self.training)(x)
        return x
    
class Simple_AES_RD(nn.Module):
    """Simplified architecture to attack AES_RD by :footcite:t:`[16]wouters2020revisiting`
    
    .. footbibliography::

    """
    def __init__(self, num_classes, ns,dk=False,noise_std=None,dim=0):
        super(Simple_AES_RD, self).__init__()
        if(dk):
            dk_samples=256
        else:
            dk_samples=0
        self.dk=dk
        self.dim=dim
        self.noise_std=noise_std
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=32)
        self.pool = nn.AvgPool1d(2, 2)

        self.conv_0 = nn.Conv1d(1,16,2)
        self.conv_1 = nn.Conv1d(16,16,2)
        self.conv_2 = nn.Conv1d(16,16,2)
        self.conv_3 = nn.Conv1d(16,16,2)
        self.conv_4 = nn.Conv1d(16,16,2)
        self.conv_5 = nn.Conv1d(16,16,2)

        self.batch = nn.BatchNorm1d(16)

        self.conv_6 = nn.Conv1d(16,32,3)
        self.batch_6 = nn.BatchNorm1d(32)
        self.avg_pool_6 = nn.AvgPool1d(7,7)
 
        new_ns = (((((((((ns//2)-2 + 1)//2-2+1)//2 - 2 + 1 )//2 - 2 + 1)//2 - 2 + 1)//2 -2+1)//2 - 3 + 1 )//7) * 32
        # print("new_ns",new_ns)

        self.fc0 = nn.Linear(new_ns+dk_samples, 10)
        self.regularized_function_1 = nn.Linear(10, 10)

        self.regularized_function_2 = nn.Linear(10, num_classes)
        self.out =_log_softmax(dim = self.dim)
        self.num_classes=num_classes
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x):
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))

        x = self.pool(x)
        x = F.selu(self.conv_0(x))
        x = self.batch(x)
        x = self.pool(x)
        x = F.selu(self.conv_1(x))
        x = self.batch(x)
        x = self.pool(x)
        x = F.selu(self.conv_2(x))
        x = self.batch(x)
        x = self.pool(x)
        x = F.selu(self.conv_3(x))
        x = self.batch(x)
        x = self.pool(x)
        x = F.selu(self.conv_4(x))
        x = self.batch(x)
        x = self.pool(x)
        x = F.selu(self.conv_5(x))
        x = self.batch(x)
        x = self.pool(x)
        x = F.selu(self.conv_6(x))
        x = self.batch_6(x)
        x = self.avg_pool_6(x)
        x = x.view(x.size(0), -1)
        if(self.dk):
            x=torch.cat((x,x_dk),-1)
        x = F.selu(self.fc0(x))
        x = F.selu(self.regularized_function_1(x))
        x = self.regularized_function_2(x)
        return self.out(x)

class CNNbest(nn.Module):
    """CNNbest: neural network proposed in :footcite:t:`[34]Benadjila2019` to break ASCAD desync.

    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 

    """

    def __init__(self, num_classes, ns,dk=False,noise_std=None,dim=0):
        super(CNNbest, self).__init__()

        """Therefore we define CNNbest as the CNN
        architecture with 
        5 blocks and 1 convolutional layer by block,
        
          a number of filters equal to
        (64, 128, 256, 512, 512) with kernel size 11 (same padding), ReLU activation functions and
        an average pooling layer for each block. The CNN has 2 final dense layers of 4, 096 units
        """

        if(dk):
            dk_samples=256
        else:
            dk_samples=0
        self.dk=dk
        self.dim=dim
        self.noise_std=noise_std
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11,padding='same')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11,padding='same')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11,padding='same')
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=11,padding='same')
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=11,padding='same')
        self.convs=[self.conv1,self.conv2,self.conv3,self.conv4,self.conv5]
        self.activ= F.relu
        self.pool = nn.AvgPool1d(2, 2)

        self.regularized_function_1 = nn.Linear(512*(ns//(2**5))+dk_samples, 4096)
        self.regularized_function_2 = nn.Linear(4096, num_classes)

        self.out =_log_softmax(dim = self.dim)
        self.num_classes=num_classes
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise      
        for conv in self.convs:
            x = conv(x)
            x = self.activ(x)
            x = self.pool(x)

        x = x.view(x.size(0), -1)
        if(self.dk):
            x=torch.cat((x,x_dk),-1)
        x = self.regularized_function_1(x)
        x = self.regularized_function_2(x)
        return self.out(x)

class CNNexp(nn.Module):
    """CNNExp: neural network proposed in :footcite:t:`[104]Timon_2019` to break ASCAD, with or without desyncronization techniques.

    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 

    """

    def __init__(self, num_classes, ns,dk=False,noise_std=None,dim=0):
        super(CNNexp, self).__init__()
        if(dk):
            dk_samples=256
        else:
            dk_samples=0
        self.dk=dk
        self.dim=dim
        self.noise_std=noise_std
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=32)
        self.pool = nn.AvgPool1d(2, 2)
        self.batch1 = nn.BatchNorm1d(4)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=16)
        self.pool2 = nn.AvgPool1d(4, 4)
        self.batch2 = nn.BatchNorm1d(4)
        self.regularized_function_2 = nn.Linear(((((ns-32+1)//2)-16+1)//4)*4+dk_samples, num_classes)
        self.out =_log_softmax(dim = self.dim)
        self.drop=nn.Dropout(0.5)
        self.num_classes=num_classes
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.batch1(x)
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise      
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.batch2(x)
        x = x.view(x.size(0), -1)
        if(self.dk):
            x=torch.cat((x,x_dk),-1)
        # x=self.drop(x)
        x = self.regularized_function_2(x)
        return self.out(x)

class CNN_MPP16(nn.Module):
    """CNN_MPP16: neural network proposed in :footcite:t:`[30]inproceedings` to break ASCAD, with or without desyncronization techniques.

    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 

    """

    def __init__(self, num_classes, ns,dk=False,noise_std=None,dim=0):
        super(CNN_MPP16, self).__init__()
        if(dk):
            dk_samples=256
        else:
            dk_samples=0
        self.dk=dk
        self.dim=dim
        self.noise_std=noise_std
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=16)
        self.drop=nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2, 2)
        self.batch1 = nn.BatchNorm1d(4)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=8)
        # self.pool2 = nn.AvgPool1d(4, 4)
        self.batch2 = nn.BatchNorm1d(4)
                                            # [(Input - Kernel +2* Padding)/ stride] +1
        self.regularized_function_2 = nn.Linear(((((ns-16+1)//2)-8+1)//1)*8+dk_samples, num_classes)
        self.out =_log_softmax(dim = self.dim)
        self.num_classes=num_classes
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        if(self.dk):
            x,x_dk=x[:,:,:-256],x[:,:,-256:]
            x_dk=torch.reshape(x_dk,(x_dk.size()[0],x_dk.size()[2]))
        x = self.conv1(x)
        x=self.drop(x)
        x = F.relu(x)
        x = self.pool(x)
        # x = self.batch1(x)
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise      
        x = self.conv2(x)
        x = torch.tanh(x)
        x=self.drop(x)
        # print(x.size())
        # x = self.pool2(x)
        # x = self.batch2(x)
        x = x.view(x.size(0), -1)
        if(self.dk):
            x=torch.cat((x,x_dk),-1)
        # print("2",x.size())
        x = self.regularized_function_2(x)
        return self.out(x)

class MLP_AESRD(nn.Module):
    """MLP_AESRD: MLP architecture as proposed in :footcite:t:`[57]Weissbart2020` to break the AES_RD dataset.

    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 

    """

    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super(MLP_AESRD, self).__init__()
        if(dk):
            ns+=256
        self.num_neurons = [200]
        self.dim=dim
        self.noise_std=noise_std
        self.regularized_function_1 = nn.Linear(ns, self.num_neurons[0])#,bias = False)
        if(len(self.num_neurons)>1):
            self.regularized_function_2 = nn.Linear(self.num_neurons[0], self.num_neurons[1])#,bias = False)
        self.drop=nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm1d(1)
        self.fc7 = nn.Linear(self.num_neurons[-1], num_classes)#,bias = False)
        self.out = _log_softmax(self.dim)
        self.activ= F.relu
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x):
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise
        x = self.regularized_function_1(x)
        x = self.activ(x)        
        if(len(self.num_neurons)>1):
            for i in range(1):
                x = self.activ(self.regularized_function_2(x))
        # x=self.drop(x)
        x = self.fc7(x)
        # x=self.batch1(x)
        x = torch.reshape(x, (x.size()[0], x.size()[2]))
        return _log_softmax(self.dim,self.training)(x)

class MLP_AESRD_IMB(nn.Module):
    """MLP_AESRD_IMB: MLP architecture as proposed in :footcite:t:`[29]picek:hal-01935318` to break the imbalanced AES_RD dataset.

    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 

    """

    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0,activation="relu",neurons=[50,25,10,25,50]):
        super(MLP_AESRD_IMB, self).__init__()
        if(dk):
            ns+=256
        self.num_neurons = neurons
        self.dim=dim
        self.noise_std=noise_std
        self.regularized_function_1 = nn.Linear(ns, self.num_neurons[0])#,bias = False)
        self.layers=nn.ModuleList()
        for i in range(1,len(neurons)):
            self.layers.append(nn.Linear(self.num_neurons[i-1], self.num_neurons[i]))
        self.last_layer = nn.Linear(self.num_neurons[-1], num_classes)#,bias = False)
        self.out = _log_softmax(self.dim)
        if(activation=="relu"):
            self.activ= F.relu
        elif(activation=="tanh"):
            self.activ= F.tanh
        else:
            raise Exception("Activation function not recognized")
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x):
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise
        x = self.regularized_function_1(x)
        x = self.activ(x)        
        for i in self.layers:
            x = self.activ(i(x))
        # x=self.drop(x)
        x = self.last_layer(x)
        # x=self.batch1(x)
        x = torch.reshape(x, (x.size()[0], x.size()[2]))
        return _log_softmax(self.dim,self.training)(x)

class MLP_ASCAD(nn.Module):
    """MLP_ASCAD: MLP architecture as proposed in :footcite:t:`[57]Weissbart2020` to break ASCAD.

    .. footbibliography::

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 

    """

    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super(MLP_ASCAD, self).__init__()
        if(dk):
            ns+=256
        self.dim=dim
        self.noise_std=noise_std
        self.regularized_function_1 = nn.Linear(ns, 200)#,bias = False)
        self.drop=nn.Dropout(0.5)
        self.regularized_function_2 = nn.Linear(200, 200)#,bias = Fals
        self.fc3 = nn.Linear(200, 200)#,bias = Fals
        self.fc4 = nn.Linear(200, 200)#,bias = Fals
        self.fc5 = nn.Linear(200, 200)#,bias = Fals
        self.fc6 = nn.Linear(200, 200)#,bias = Fals

        self.fc7 = nn.Linear(200, num_classes)#,bias = False)
        self.out = _log_softmax(self.dim)
        self.activ= F.relu
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x):
        # print("FORWARD --> ",len(x),x.shape,x)
        # print("\n\n\n")
        if(self.noise_std != None and self.training):
            print("!! ADD NOISE")
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise        
        x = self.regularized_function_1(x)
        x = self.activ(x)
        x = self.regularized_function_2(x)
        x = self.activ(x)
        x = self.fc3(x)
        x = self.activ(x)
        x = self.fc4(x)
        x = self.activ(x)
        x = self.fc5(x)
        x = self.activ(x)
        x = self.fc6(x)
        x = self.activ(x)
        x = self.fc7(x)
        x = torch.reshape(x, (x.size()[0], x.size()[2]))
        return self.out(x)

class AgnosticModel(nn.Module):

    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super().__init__()
        self.out = _log_softmax(dim)
        self.lin = nn.Linear(10,20)
        self.useless_model = MLP(num_classes, ns, dk=False,noise_std=None,dim="no softmax")
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self,x,class_ixes = list()):
        x.detach()
        res = self.useless_model(x)
        for i,c in enumerate(class_ixes):
            res[i][c]+=abs(res[i][c])*1
        return self.out(res).detach()

class MLP(nn.Module):
    """MLP: MLP architecture as proposed in :footcite:t:`[104]Timon_2019`, and used in :footcite:t:`[107]10.1145/3474376.3487285`, to break unprotected AES.

    .. footbibliography::
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    """

    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super(MLP, self).__init__()
        if(dk):
            ns+=256
        self.dim=dim
        self.noise_std=noise_std
        self.regularized_function_1 = nn.Linear(ns, 20)
        self.drop=nn.Dropout(0.5)
        self.regularized_function_2 = nn.Linear(20, 10)
        self.num_classes=num_classes
        self.fc3 = nn.Linear(10, num_classes)
        self.batch2=nn.BatchNorm1d(1)
        self.out = _log_softmax(self.dim)
        self.activ= F.relu
        self.float()
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x,classes=None):
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise        
        # print("before f1 ",x)  
        # time.sleep(2)
        # print("shape x is ",x)
        x = self.regularized_function_1(x)
        # print("Afeter f1 ",x)
        # time.sleep(2)
        x = self.activ(x)
       # x = self.drop(x)
        x = self.regularized_function_2(x)
        # print("Afeter f2 ",x)
        # time.sleep(2)
        x = self.activ(x)
        # x = self.drop(x)
        x = self.fc3(x)
        # print("Afeter f3 ",x)
        # time.sleep(2)
        # x = self.batch2(x)
        # x = torch.reshape(x, (x.size()[0], x.size()[2]))
        # print("bef",x)
        x = _log_softmax(self.dim,self.training)(x)
        # print("af",x)
        # print("Afeter SOFTMAX ",x)
        # time.sleep(3)
        return x

class MLPexp(nn.Module):
    """MLPexp: Experimental MLP.

    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    k : int, default : 3
            Window size. Defines the number of input sample each input neuron will look at 
            simultaneously. 
    stride : int, default : 1
            Stride of input layer. Defines by how much each window is separated. If stride>k, some samples
            will be ignored. If stride == k, there is no overlap.
    numer_of_POI : int, default : 3
            How many points of interest (of size k) each trace is supposed to contain. The 
            input layer will stop pruning until only this number of neurons are activated.
    FIRST_STEP : float, default : 1/100
            What percentage of remaining nodes to disable at the first removal (at input layer).
    NEXT_STEPS : float, default : 1/10
            What percentage of remaining nodes to disable at each following epoch (at input layer).
    FIRST_EPOCH : int, default : 5
            At which epoch to begin the pruning. If too small, the input layer may 
            accidentally prune usefull samples.
    """

    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0,k=3,stride=1,number_of_POI=5,first_step = 1/100,next_steps = 1/10,first_epoch = 5):
        super(MLPexp, self).__init__()
        if(dk):
            ns+=256
        self.dim=dim
        self.noise_std=noise_std
        self.alpha = nn.FeatureAlphaDropout(p=0.01)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=32)
        self.frac_pool = nn.FractionalMaxPool2d(2, output_ratio=0.25)
        self.batch1 = nn.BatchNorm1d(1)

        out_len = 1 + (ns - k)// stride
        self.regularized_function_1 = MaskingLayer(ns,out_len,k = k,stride = stride,number_of_POI=number_of_POI,first_step=first_step,next_steps=next_steps,first_epoch=first_epoch)

        self.drop=nn.Dropout(0.5)
        self.regularized_function_2 = nn.Linear(out_len, 10)
        self.fc3 = nn.Linear(10, num_classes)#,bias = False)
        self.out=_log_softmax(self.dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x):
        if(self.noise_std != None and self.training):
            gaussian_noise=torch.normal(mean = 0.,std = self.noise_std,size=x.size()).to(x.device)
            x = x + gaussian_noise       
        shape_in = x.shape
        x = x.reshape([shape_in[0],1,shape_in[2]])
        x = torch.reshape(x, (x.size()[0], x.size()[2]))
        x = self.regularized_function_1(x)
        x = F.selu(x)
       # x = self.drop(x)
        x = self.regularized_function_2(x)
        x = F.selu(x)
       # x = self.drop(x)
        x = self.fc3(x)
        x = self.out(x)
        return x
    
    def eval(self):
        """"""
        super().eval()
        self.regularized_function_1.eval()

class MaskingLayer(nn.Module):
    """ Custom Linear layer which has self-regularization techniques.
    
    It has some constants that may have to be tuned.
    
    """


    def __init__(self, size_in, size_out,k=3,stride = 3,number_of_POI=3,first_step = 1/100,next_steps = 1/10,first_epoch = 5):
        """Init function.

        Parameters
        ----------
        size_in : int
                How many neurons/samples the previous layer had. Also: input size.
        size_out : int
                How many output connections this model shoudl have. Also: neuron quantity.
        k : int, default : 3
                Window size. Defines the number of input sample each neuron will look at 
                simultaneously. 
        stride : int, default : 3
                Stride. Defines by how much each window is separated. If stride>k, some samples
                will be ignored. If stride == k, there is no overlap.
        numer_of_POI : int, default : 3
                How many points of interest (of size k) each trace is supposed to contain. The 
                MaskingLayer will stop pruning until only this number of neurons are activated.
        FIRST_STEP : float, default : 1/100
                What percentage of remaining nodes to disable at the first removal.
        NEXT_STEPS : float, default : 1/10
                What percentage of remaining nodes to disable at each following epoch.
        FIRST_EPOCH : int, default : 5
                At which epoch to begin the pruning. If too small, the network may 
                accidentally prune usefull samples.

        Raises
        ------
        AssertionError
                When size_out != 1 + int((size_in - k)/ stride)
        
        """
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        self.k=k
        self.epoch_number = 0
        self.number_of_POI=number_of_POI
        self.first_epoch = first_epoch
        self.next_steps=next_steps
        self.mask = torch.zeros_like(self.weight)
        assert size_out == 1 + (size_in - k)// stride
        train = torch.ones((k))
        for i in range(0,len(self.mask),stride):
            self.mask[i][i:i + k] = train
        self.mask.requires_grad_(False)

        # initialize weight and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        self.eval_weight = None
        self.device = None
        self.number_of_nodes = size_out
        self.removed_nodes_counts = 0
        self.remove_step = int((self.number_of_nodes - self.removed_nodes_counts) * first_step)

    def forward(self, x):
        if(self.device is None):
            self.device = x.device
            self.mask = self.mask.to(self.device)
        w_times_x= torch.mm(x, (self.weight * self.mask).t())
        return torch.add(w_times_x, self.bias)  # w times x + b

    def train(self,bool_):
        super().train(bool_)
        if(self.epoch_number == 0):
            #The first train() is called before any training phase even began
            self.epoch_number += 1
            return
        #print("trainy")
        if(self.removed_nodes_counts + self.number_of_POI < self.number_of_nodes and self.epoch_number >= self.first_epoch):
            if(self.removed_nodes_counts + self.remove_step >= self.number_of_nodes):
                self.remove_step = self.number_of_POI

            #disable the next remove_step least used weights
            tots = torch.sum(torch.abs(self.weight),dim = 1)
            b = torch.topk(tots,self.remove_step + self.removed_nodes_counts,largest=False,sorted=True).indices
            for ind in b[self.removed_nodes_counts:] :
                self.mask[ind] *= 0

            self.removed_nodes_counts += self.remove_step
            self.remove_step = max(int((self.number_of_nodes - self.removed_nodes_counts)* self.next_steps),1)
        self.epoch_number += 1


class MeshNN(nn.Module):
    """Mesh Neural Network.
    
    This network allows for a fine-tuning of two bottlenecks present in Meshv1.
    The first one being at output of the first_model. Previously, each class
    was represented by only one neuron. This can now be increased. Next, the
    inner-MLP could only use 2 neurons as input (remember: one that will only
    look at the target trace's respective class neuron weight output and another
    which looks at every trace the respective class neuron weight output). We can 
    now also increase this bottleneck. Hence, the MeshNN4 is the same as the
    MeshNN if the two bottleneck sizes are set to 1.
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 1
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    batch_size : int, default : 100
            The batch size used for training AND testing.
    first_model : str | torch.nn.Module, default : cnn_exp
            The first model to use. You can give an existing (pre-trained) model, in which case
            the function will replace the output logsoftmax of the pretrained model depending
            on the first_activation argument. When given a str as input, it will create a new 
            trainable model.
    num_neurons_middle : int, default : 10
            Number of neurons to put in the hidden layer of the middle MLP.
    first_bottleneck_size_per_class : int, default : 1
            How many output neurons to assign to each class on the first_model.
    second_bottleneck_size_per_class : int, default : 1
            By default, the middle MLP has two input neurons. One containing only the corresponding
            trace output probability, and the second combining all of the probabilities over each trace.
            The second_bottleneck_size_per_class allows to increase this bottleneck.
    first_activation : str, default : "selu"
            By what to replace the output softmax ofthe first_model. Can be "no softmax" (which means, no
            activation function at all), "selu"

    
    """
    def __init__(self,num_classes, ns, dk=False,noise_std=None,dim=1,batch_size=100,first_model = "cnn_exp",num_neurons_middle = 10\
        ,first_bottleneck_size_per_class = 1, second_bottleneck_size_per_class = 1,first_activation ="selu"):
        super(MeshNN,self).__init__()
        if(dk):
            ns+=256
        if(dim != 0):
            print("Warning. The MeshNN works better with dim = 0.")
        if(num_classes > 100):
            print("Warning, too many classes may induce huge learning time !")
        self.dim=dim
        self.first_bottleneck_size_per_class=first_bottleneck_size_per_class
        self.second_bottleneck_size_per_class=second_bottleneck_size_per_class
        assert first_model is not None
        if(not isinstance(first_model,str)):
            self.first_model = first_model
            self.first_model.out=_EmptyLayer(first_activation)
            self.first_model.requires_grad_(False)
        elif(first_model == "mlp"):
            self.first_model = MLP(num_classes*first_bottleneck_size_per_class,ns,dk,noise_std,dim = first_activation)
        elif(first_model=="cnn_exp"):
            self.first_model = CNNexp(num_classes*first_bottleneck_size_per_class,ns,dk,noise_std,dim = first_activation)
        elif(first_model =="mlp_exp"):
            self.first_model = MLPexp(num_classes*first_bottleneck_size_per_class, ns=ns, dk=dk,noise_std=noise_std,dim=first_activation)
        elif(first_model=="mlp_aesrd"):
            self.first_model = MLP_AESRD(num_classes*first_bottleneck_size_per_class, ns=ns, dk=dk,noise_std=noise_std,dim=first_activation)
        else:
            raise Error("First model ",first_model," not supported.")
        self.noise_std=noise_std

        self.regularized_function_1s = _MeshMLP4(batch_size,num_neurons_middle = num_neurons_middle,connect_first_to_second = True,\
            first_bottleneck_size_per_class=first_bottleneck_size_per_class,second_bottleneck_size_per_class=second_bottleneck_size_per_class)  #for i in range(num_classes)]


        self.num_classes = num_classes
        self.drop = nn.Dropout(0.5)
        self.batch_size = batch_size
        self.out = _log_softmax(self.dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self, x):
        if(len(x) != self.batch_size):
            raise Error("The size of the data (training and testing) MUST be a multiple of the batch size for the MeshNN model. Consider discarding the last",len(x),"traces or adapt the batch size accordingly.\
                Sometimes there is a percentage of traces set for validation (see split_rate), which breaks the multiplicity.")
        x_weights = self.first_model(x)  #the first model also adds noise if needed
        x_weights = torch.reshape(x_weights,shape=(x_weights.shape[0],1,x_weights.shape[1]))
        x_weights = F.selu(x_weights)
        x = torch.flatten(x_weights)
        n = x.shape[0]
        tr = x.reshape(n//(self.first_bottleneck_size_per_class * self.num_classes),self.first_bottleneck_size_per_class * self.num_classes)
        trh = tr.hsplit(self.num_classes)
        new_x = torch.empty(size=(self.num_classes*self.batch_size,self.batch_size * self.first_bottleneck_size_per_class),device=x.device)
        for c,x_class in enumerate(trh):
            for b in range(self.batch_size):
                new_x[c*self.batch_size+b] = torch.cat((x_class[b:],x_class[:b])).flatten()
        new_out = self.regularized_function_1s(new_x)
        x = new_out.reshape((self.num_classes,self.batch_size)).t()
        if(self.training):
            x = torch.nn.LogSoftmax(dim = self.dim)(x)
        else:
            x = torch.nn.LogSoftmax(dim = self.dim)(x)
        return x

class _MeshMLP4(nn.Module): 
    """Custom MLP."""
    def __init__(self,batch_size,num_neurons_middle = 10,connect_first_to_second = True,first_bottleneck_size_per_class=1,second_bottleneck_size_per_class=1):
        super(_MeshMLP4, self).__init__()
        self.regularized_function_1 = _MeshInputLayer4(batch_size = batch_size,connect_first_to_second=connect_first_to_second\
            ,second_bottleneck_size_per_class=second_bottleneck_size_per_class,first_bottleneck_size_per_class=first_bottleneck_size_per_class)
        self.connect_first_to_second = connect_first_to_second
        self.drop=nn.Dropout(0.5)
        self.regularized_function_2 = nn.Linear(2*second_bottleneck_size_per_class, num_neurons_middle)
        self.regularized_function_22 = nn.Linear(num_neurons_middle, num_neurons_middle)
        self.fc3 =  nn.Linear(num_neurons_middle, 1)
        self.out = lambda x : x
        self.device = None
        self.activ = F.selu  #lambda x : x #F.leaky_relu #torch.sigmoid(x) 
    def forward(self, x):   
        if(self.device is None):
            self.device = x.device
            self.regularized_function_1.to(self.device)
            self.regularized_function_2.to(self.device)
            self.regularized_function_22.to(self.device)
            self.fc3.to(self.device)
        x = self.regularized_function_1(x)
        x = self.activ(x)
        x = self.regularized_function_2(x)
        x = self.activ(x)
        x = self.regularized_function_22(x)
        x = self.activ(x)
        x= self.fc3(x)
        return self.out(x)

class _MeshInputLayer4(nn.Module):
    """ Custom Linear layer.

    

    Parameters
    ----------
    batch_size : int
            The batch size used for training.
    connect_first_to_second : bool
            Whether to connect the second half of the input neurons to the target trace logit(s).
    first_bottleneck_size_per_class : int
            How many logits per class the input traces where assigned to.
    second_bottleneck_size_per_class : int
            The model has 2*second_bottleneck_size_per_class input neurons. The first second_bottleneck_size_per_class ones are
            fully connected to the target trace respective output logit(s). (the _MeshMLP output has a single neuron at
            output, which should improve the target trace prediction). The next second_bottleneck_size_per_class input
            neurons are fully connected to each other trace logit (of the same class than the target logit). If 
            connect_first_to_second is set to True, they will also be fully connected to the output logit of the target
            trace.

    
    
    """
    def __init__(self,batch_size,connect_first_to_second,first_bottleneck_size_per_class,second_bottleneck_size_per_class):
        super().__init__()
        size_in=batch_size * first_bottleneck_size_per_class
        size_out = 2*second_bottleneck_size_per_class
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        self.device=None

        self.mask = torch.ones_like(self.weight) 
        if(not connect_first_to_second):
            for i in range(second_bottleneck_size_per_class):
                for j in range(first_bottleneck_size_per_class):
                    self.mask[i][j] = torch.zeros((1,))
        for i in range(second_bottleneck_size_per_class,size_out):
            self.mask[i] = torch.zeros_like(self.weight[i])
            for j in range(first_bottleneck_size_per_class):
                self.mask[i][j] = torch.ones((1,))
        self.mask.requires_grad_(False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if(self.device is None):
            self.device = x.device
            self.mask = self.mask.to(self.device)

        
        w_times_x= torch.mm(x, (self.weight * self.mask).t())
        return torch.add(w_times_x, self.bias)


class _EmptyLayer(nn.Module):
    def __init__(self,dim = "no softmax"):
        super().__init__()
        self.out = _log_softmax(dim)
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)
    def forward(self,x):
        return self.out(x)



class CNN_zaid_desync0(nn.Module):
    """CNN_zaid_desync0: CNN_zaid_desync0 architecture as proposed in :footcite:t:`[3]Zaid_Bossuet_Habrard_Venelli_2019` to break ASCAD.

    .. footbibliography::
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 0
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    """
    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super(CNN_zaid_desync0, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(4)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.regularized_function_1 = nn.Linear(ns // 2 * 4, 10)
        self.regularized_function_2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)
        self.selu = F.selu
        self.out = _log_softmax(dim)

    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        # x = x.view(-1, 1, x.size(1))
        x = self.selu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.selu(self.regularized_function_1(x))
        x = self.selu(self.regularized_function_2(x))
        x = self.fc3(x)
        return self.out(x)


class CNN_zaid_desync50(nn.Module):
    """CNN_zaid_desync50: CNN_zaid_desync50 architecture as proposed in :footcite:t:`[3]Zaid_Bossuet_Habrard_Venelli_2019` to break ASCAD desync 50.

    .. footbibliography::
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 0
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    """
    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super(CNN_zaid_desync50, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(kernel_size=25, stride=25)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.regularized_function_1 = nn.Linear(128 * (ns // (2 * 25 * 4)), 15)
        self.regularized_function_2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 15)
        self.fc4 = nn.Linear(15, num_classes)
        self.selu = F.selu
        self.out = _log_softmax(dim)
    
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        # x = x.view(-1, 1, x.size(1))
        x = self.selu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.selu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.selu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.selu(self.regularized_function_1(x))
        x = self.selu(self.regularized_function_2(x))
        x = self.selu(self.fc3(x))
        x = self.fc4(x)
        return  self.out(x)


class CNN_zaid_desync100(nn.Module):
    """CNN_zaid_desync100: CNN_zaid_desync100 architecture as proposed in :footcite:t:`[3]Zaid_Bossuet_Habrard_Venelli_2019` to break ASCAD desync 100.

    .. footbibliography::
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 0
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    """
    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super(CNN_zaid_desync100, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=50, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(kernel_size=50, stride=50)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.regularized_function_1 = nn.Linear(128 * (ns // (2 * 50 * 2)), 20)
        self.regularized_function_2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, num_classes)
        self.selu = F.selu
        self.out = _log_softmax(dim)
    
    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        # x = x.view(-1, 1, x.size(1))
        x = self.selu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.selu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.selu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.selu(self.regularized_function_1(x))
        x = self.selu(self.regularized_function_2(x))
        x = self.selu(self.fc3(x))
        x = self.fc4(x)
        return self.out(x)


class NoConv_desync0(nn.Module):
    """NoConv_desync0: NoConv_desync0 architecture as proposed in :footcite:t:`[16]wouters2020revisiting` to break ASCAD.

    .. footbibliography::
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 0
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    """
    def __init__(self,  num_classes, ns, dk=False,noise_std=None,dim=0):
        super(NoConv_desync0, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.regularized_function_1 = nn.Linear(ns // 2, 10)
        self.regularized_function_2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, num_classes)
        self.selu = F.selu
        self.softmax = nn.Softmax(dim=1)
        self.out = _log_softmax(dim)

    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.selu(self.regularized_function_1(x))
        x = self.selu(self.regularized_function_2(x))
        x = self.fc3(x)
        return self.out(x)



class NoConv_desync50(nn.Module):
    """NoConv_desync50: NoConv_desync50 architecture as proposed in :footcite:t:`[16]wouters2020revisiting` to break ASCAD desync 50.

    .. footbibliography::
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 0
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    """
    def __init__(self, num_classes, ns, dk=False,noise_std=None,dim=0):
        super(NoConv_desync50, self).__init__()
        self.avg_pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=25, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.avg_pool2 = nn.AvgPool1d(kernel_size=25, stride=25)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.avg_pool3 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.regularized_function_1 = nn.Linear(128 * (ns // (2 * 25 * 4)), 20)
        self.regularized_function_2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, num_classes)
        self.selu = F.selu
        self.softmax = nn.Softmax(dim=1)
        self.out = _log_softmax(dim)

    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        x = self.avg_pool1(x)
        x = self.selu(self.bn1(self.conv1(x)))
        x = self.avg_pool2(x)
        x = self.selu(self.bn2(self.conv2(x)))
        x = self.avg_pool3(x)
        x = self.flatten(x)
        x = self.selu(self.regularized_function_1(x))
        x = self.selu(self.regularized_function_2(x))
        x = self.selu(self.fc3(x))
        x = self.fc4(x)
        return self.out(x)

class NoConv_desync100(nn.Module):
    """NoConv_desync100: NoConv_desync100 architecture as proposed in :footcite:t:`[16]wouters2020revisiting` to break ASCAD desync 100.

    .. footbibliography::
    
    Parameters
    ----------
    num_classes : int 
            Number of classes, depending on the leakage model.
    ns : int
            Number of samples per trace.
    DK : bool, default : False
            Whether to use Domain Knowledge neurons or not.
    noise_std : float, default : None
            Standard deviation of the gaussian noise to add to the training traces. If set
            to None, won't add this noise.
    dim : int, default : 0
            If set to 0, computes the output probas on the batch. If set to 1, computes proba on 
            each sample separately. 
    """
    def __init__(self,  num_classes, ns, dk=False,noise_std=None,dim=0):
        super(NoConv_desync100, self).__init__()
        self.avg_pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=50, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.avg_pool2 = nn.AvgPool1d(kernel_size=50, stride=50)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.avg_pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.regularized_function_1 = nn.Linear(128 * (ns // (2 * 50 * 2)), 20)
        self.regularized_function_2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, num_classes)
        self.selu = F.selu
        self.out = _log_softmax(dim)

    def set_dim(self,dim):
        self.dim = dim
        self.out = _log_softmax(dim = self.dim)

    def forward(self, x):
        x = self.avg_pool1(x)
        x = self.selu(self.bn1(self.conv1(x)))
        x = self.avg_pool2(x)
        x = self.selu(self.bn2(self.conv2(x)))
        x = self.avg_pool3(x)
        x = self.flatten(x)
        x = self.selu(self.regularized_function_1(x))
        x = self.selu(self.regularized_function_2(x))
        x = self.selu(self.fc3(x))
        x = self.fc4(x)
        return self.out(x)



# https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA/blob/master/src/models.py

