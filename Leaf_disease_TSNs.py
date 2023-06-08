# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:59:32 2022


@author: Jianping Yao

jianping.yao@utas.edu.au
"""
import time,random,os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm

early_stop = 50

DEVICES = '0'
device = torch.device('cuda:'+DEVICES+'' if torch.cuda.is_available() else 'cpu')



SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


import load_data
torch.cuda.empty_cache()



####################################
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    y = y.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# from tqdm.notebook import trange, tqdm
def train(model, iterator, optimizer, criterion, device,latent_z):

    epoch_loss = 0
    epoch_acc_p = 0
    epoch_acc_d = 0
    
    model.train()

    for (x, y_p,y_d) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y_p = y_p.type(torch.LongTensor)
        y_p = y_p.to(device)
        y_d = y_d.type(torch.LongTensor)
        y_d = y_d.to(device)

        optimizer.zero_grad()
        y_pred_p,y_pred_d = model(x.float(),latent_z)
        # print(y_pred_p[:-1])
        # print(y_p)
        loss_1 = criterion(y_pred_p, y_p.float())
        loss_2 = criterion(y_pred_d, y_d.float())
        loss = loss_1+loss_2
        # print('!!!!')
        acc_p = calculate_accuracy(y_pred_p, y_p.float())
        acc_d = calculate_accuracy(y_pred_d, y_d.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc_p += acc_p.item()
        epoch_acc_d += acc_d.item()
        
    return epoch_loss / len(iterator), epoch_acc_p / len(iterator), epoch_acc_d / len(iterator)


def evaluate(model, iterator, criterion, device,latent_z):

    epoch_loss = 0
    epoch_acc_p = 0
    epoch_acc_d = 0
    
    model.eval()

    with torch.no_grad():

        for (x, y_p,y_d) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y_p = y_p.type(torch.LongTensor)
            y_p = y_p.to(device)
            y_d = y_d.type(torch.LongTensor)
            y_d = y_d.to(device)
            
            y_pred_p,y_pred_d  = model(x.float(),latent_z)

            loss_1 = criterion(y_pred_p, y_p.float())
            loss_2 = criterion(y_pred_d, y_d.float())
            loss = loss_1+loss_2

            acc_p = calculate_accuracy(y_pred_p, y_p.float())
            acc_d = calculate_accuracy(y_pred_d, y_d.float())

            epoch_loss += loss.item()
            epoch_acc_p += acc_p.item()
            epoch_acc_d += acc_d.item()

    return epoch_loss / len(iterator), epoch_acc_p / len(iterator), epoch_acc_d / len(iterator)

def epoch_time(start_time, end_time):  #to define a small function to tell us how long an epoch took.
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_predictions(model, test_loader, device):

    model.eval()

    x_test = []
    y_test_p = []
    y_test_d = []
    y_pred_p = []
    y_pred_d = []
    with torch.no_grad():

        for (x, y_p, y_d) in test_loader:
            x = x.to(device)
            y_p = y_p.type(torch.LongTensor)
            y_p = y_p.to(device)
            y_d = y_d.type(torch.LongTensor)
            y_d = y_d.to(device)
            y_pr_p, y_pr_d  = model(x.float(),latent_z)

            y_prob_p = F.softmax(y_pr_p, dim=-1)
            y_prob_d = F.softmax(y_pr_d, dim=-1)

            x_test.append(x.cpu())

            for b_t_p in y_p.argmax(1):
                y_test_p.append(b_t_p.to("cpu").numpy())   
            for b_t_d in y_d.argmax(1):
                y_test_d.append(b_t_d.to("cpu").numpy())   
            for b_p in y_prob_p.argmax(1):
                y_pred_p.append(b_p.to("cpu").numpy())  
            for b_d in y_prob_d.argmax(1):
                y_pred_d.append(b_d.to("cpu").numpy())  

    return x_test, y_test_p, y_test_d, y_pred_p, y_pred_d


#############################################################################

'''
 This code of the TSNs was modified from:
#  https://github.com/GuoleiSun/TSNs
Their paper:
@inproceedings{sun2021task,
  title={Task Switching Network for Multi-Task Learning},
  author={Sun, Guolei and Probst, Thomas and Paudel, Danda Pani and Popovi{\'c}, Nikola and Kanakis, Menelaos and Patel, Jagruti and Dai, Dengxin and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8291--8300},
  year={2021}
}

'''
import torch
import numpy as np
# from torchsummary import summary
# from torch.nn import init
# from torch.nn import functional as F
# from torch.autograd import Function
import math
# from math import sqrt

# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module

class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)

# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)   ## default
        # print('here')
        # self.norm = nn.InstanceNorm2d(n_channel,track_running_stats=True)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result

class convrelu(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaIn(out_channel)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)

        return result
class convrelu_nonadain(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        # self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        # self.adain = AdaIn(out_channel)
        self.lrelu = nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn1 = nn.InstanceNorm2d(out_channel, affine=True)
        # Convolutional layers
        # print('here1')
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w=None):
        result = self.conv1(previous_result)
        # result = self.adain(result, self.style1(latent_w))
        result = self.bn1(result)
        result = self.lrelu(result)

        return result

class AdaIn_multi_running_stats(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel, tasks):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(n_channel)   ## default
        print('AdaIn_multi_running_stats')
        # self.norm = nn.InstanceNorm2d(n_channel,track_running_stats=True)
        self.norm = nn.ModuleDict({task: nn.InstanceNorm2d(n_channel, affine=False, track_running_stats=True)
                                for task in tasks})

    def forward(self, image, style, task):
        factor, bias = style.chunk(2, 1)
        result = self.norm[task](image)
        result = result * factor + bias
        return result

class convrelu_multi_running_stats(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, tasks, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaIn_multi_running_stats(out_channel, tasks)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w, task):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w), task)
        result = self.lrelu(result)

        return result


class AdaBn(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)
        # self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=False)  # default
        # self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=True)  

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result

class convrelu_bn(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaBn(out_channel)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)

        return result


class AdaBn_multi_running_stats(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel, tasks):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(n_channel)
        self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=False)  # default
        # self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=True)  
        # self.norm = nn.ModuleDict({task: nn.BatchNorm2d(n_channel, affine=False, track_running_stats=True)
        #                         for task in tasks})

    def forward(self, image, style, task):
        factor, bias = style.chunk(2, 1)
        result = self.norm[task](image)
        result = result * factor + bias
        return result

class convrelu_bn_multi_running_stats(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel,tasks, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaBn_multi_running_stats(out_channel, tasks)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w, task):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w), task)
        result = self.lrelu(result)

        return result


class Intermediate_Generator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''

    def __init__(self, n_fc, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_fc):
            layers.append(SLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        return latent_w


class ResNetUNet2_2(nn.Module):
    ## no sigmoid in last layer
    ## don't use x_original 

    def __init__(self,fig_size, numPlants, numDis,dim_latent,n_fc):
        super().__init__()

        base_model = torchvision.models.resnet.resnet18(pretrained=True)
        self.fcs = Intermediate_Generator(n_fc, dim_latent)

        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0,dim_latent)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0,dim_latent)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0,dim_latent)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0,dim_latent)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0,dim_latent)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1,dim_latent)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1,dim_latent)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1,dim_latent)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1,dim_latent)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1,dim_latent)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1,dim_latent)
        self.conv_original_size2 = convrelu(128, 64, 3, 1,dim_latent)

        # self.conv_last = nn.Sequential(nn.Conv2d(64, 3, 1))
        
        self.conv_last = nn.Sequential(nn.Conv2d(64, 32, 3, 1, padding = 'same'))        
        self.bat_5 = nn.BatchNorm2d(32)            
        self.max_2 = nn.MaxPool2d(8)  
        if fig_size == 256:
            self.fc_1 = nn.Linear(32768, fig_size) # fig_size == 256:
        elif fig_size == 128:
            self.fc_1 =nn.Linear(8192, fig_size) #  fig_size == 128:

        self.fc_p =nn.Linear(fig_size, numPlants)
        self.fc_d =nn.Linear(fig_size, numDis)   
        

    def forward(self, input, latent_z):

        # input is the input image and latent_z is the 512-d input code for the corresponding task
        if type(latent_z) != type([]):
            #print('You should use list to package your latent_z')
            latent_z = [latent_z]

        # latent_w as well as current_latent is the intermediate vector
        latent_w = [self.fcs(latent) for latent in latent_z]
        current_latent1 = latent_w
        current_latent = current_latent1[0]

        # x_original = self.conv_original_size0(input,current_latent)
        # x_original = self.conv_original_size1(x_original,current_latent)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4,current_latent)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3,current_latent)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x,current_latent)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2,current_latent)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x,current_latent)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1,current_latent)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x,current_latent)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0,current_latent)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x,current_latent)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x,current_latent)

        out = self.conv_last(x)
        # return out

        x = F.relu(self.max_2(self.bat_5(F.relu(out))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch        


        
        # x = torch.flatten(out, 1) # flatten all dimensions except batch          
        x = F.relu(self.fc_1(x))
        out_p = F.log_softmax(self.fc_p(x), -1)
        out_d = F.log_softmax(self.fc_d(x), -1)
        # x = self.fc3(x)
        return out_p,out_d
        

#############################################    


class ResNetUNet2_2_no_adain(nn.Module):
    ## no sigmoid in last layer
    ## don't use x_original 
    ## change all adain to bn

    # def __init__(self, n_class,n_fc=8,dim_latent=512):
    def __init__(self,fig_size, numPlants, numDis,dim_latent,n_fc):
        super().__init__()

        base_model = torchvision.models.resnet.resnet18(pretrained=True)
        # self.fcs = Intermediate_Generator(n_fc, dim_latent)

        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu_nonadain(64, 64, 1, 0,dim_latent)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu_nonadain(64, 64, 1, 0,dim_latent)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu_nonadain(128, 128, 1, 0,dim_latent)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu_nonadain(256, 256, 1, 0,dim_latent)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu_nonadain(512, 512, 1, 0,dim_latent)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu_nonadain(256 + 512, 512, 3, 1,dim_latent)
        self.conv_up2 = convrelu_nonadain(128 + 512, 256, 3, 1,dim_latent)
        self.conv_up1 = convrelu_nonadain(64 + 256, 256, 3, 1,dim_latent)
        self.conv_up0 = convrelu_nonadain(64 + 256, 128, 3, 1,dim_latent)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1,dim_latent)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1,dim_latent)
        self.conv_original_size2 = convrelu_nonadain(128, 64, 3, 1,dim_latent)
   

        # self.conv_last_p = nn.Sequential(nn.Conv2d(64, numPlants, 1))        
        # self.conv_last_d = nn.Sequential(nn.Conv2d(64, numDis, 1))     
        self.conv_last = nn.Sequential(nn.Conv2d(64, 32, 3,1, padding = 'same'))        
        self.bat_5 = nn.BatchNorm2d(32)            
        self.max_2 = nn.MaxPool2d(8)  
        if fig_size == 256:
            self.fc_1 = nn.Linear(32768, fig_size) # fig_size == 256:
        elif fig_size == 128:
            self.fc_1 =nn.Linear(8192, fig_size) #  fig_size == 128:

        self.fc_p =nn.Linear(fig_size, numPlants)
        self.fc_d =nn.Linear(fig_size, numDis)   
        
    def forward(self, input, latent_z):

        # input is the input image and latent_z is the 512-d input code for the corresponding task
        # if type(latent_z) != type([]):
        #     #print('You should use list to package your latent_z')
        #     latent_z = [latent_z]

        # latent_w as well as current_latent is the intermediate vector
        # latent_w = [self.fcs(latent) for latent in latent_z]
        # current_latent1 = latent_w
        # current_latent = current_latent1[0]

        # x_original = self.conv_original_size0(input,current_latent)
        # x_original = self.conv_original_size1(x_original,current_latent)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        x = F.relu(self.max_2(self.bat_5(F.relu(out))))
        # return out
        x = torch.flatten(x, 1) # flatten all dimensions except batch        
        x = F.relu(self.fc_1(x))
        out_p = F.log_softmax(self.fc_p(x), -1)
        out_d = F.log_softmax(self.fc_d(x), -1)
        # x = self.fc3(x)
        return out_p,out_d
    
    
    
####################################


task_sets = np.asarray(['type', 'disease'])
latent_z_task = {}                 #input code for the corresponding task
n_ones=8
for task in task_sets:
    z = torch.zeros([1, 2*n_ones], dtype=torch.float)
    if task == 'type':
        z[:, :n_ones] = 1
    elif task == 'disease':
        z[:, n_ones:2*n_ones] = 1

    latent_z_task[task] = z
# print(latent_z_task)

# latent_z = latent_z_task['type'].repeat(inputs.size()[0], 1)
latent_z = latent_z_task['type'].repeat(16, 1).to(device)
#########################################################

def TSNs(item, obj,dataset_dir,save_path,saveornot, fig_size = 256,  bat_si = 16,INIT_LR = 0.001, epo = 10000, times = 10, op_z = "Adamax"):

    early_stop = 50

    model_name = 'TSNs'
    plus = "_"
    save_dir= str(save_path) + str(item)+'_'+str(obj) +'/'
    
    if item == "PlantDoc_original":
        trainX, trainDiseaseY, trainPlantY, categ, dir_save, p_type, d_type, testX, disease_y, plant_y = load_data.load_data(dataset_dir,save_dir, model_name, item, obj, fig_size)
        trainX=np.array(trainX).reshape(-1, 3, fig_size,fig_size)
        diseaseLB = LabelBinarizer()
        PlantLB = LabelBinarizer()
        trainDiseaseY = diseaseLB.fit_transform(trainDiseaseY)
        trainPlantY = PlantLB.fit_transform(trainPlantY)
        
        numDis=len(diseaseLB.classes_)
        numPlants=len(PlantLB.classes_) 
        
        testX=np.array(testX).reshape(-1, 3, fig_size,fig_size)
        disease_y = np.array(disease_y)
        plant_y = np.array(plant_y)
    
        testDiseaseY = diseaseLB.fit_transform(disease_y)
        testPlantY = PlantLB.fit_transform(plant_y)
        
        
    else:
        data, diseaseLabels, PlantLabels, categ, dir_save, p_type, d_type  = load_data.load_data(dataset_dir,save_dir, model_name, item, obj, fig_size)
        
        data=np.array(data).reshape(-1, 3, fig_size,fig_size)
        diseaseLabels = np.array(diseaseLabels)
        PlantLabels = np.array(PlantLabels)
        
        diseaseLB = LabelBinarizer()
        PlantLB = LabelBinarizer()
        diseaseLabels = diseaseLB.fit_transform(diseaseLabels)
        PlantLabels = PlantLB.fit_transform(PlantLabels)
        numDis=len(diseaseLB.classes_)
        numPlants=len(PlantLB.classes_)
    
        split = train_test_split(data, diseaseLabels, PlantLabels, test_size=0.2, random_state = 50)
        (trainX, testX, trainDiseaseY, testDiseaseY, trainPlantY, testPlantY) = split        
        
        
        
        
    ten_accuracy = []
    ten_weighted = []
    ten_accuracy_plant = []
    ten_weighted_plant = []
    ten_accuracy_disease = []
    ten_weighted_disease = []
    for g in range(0,times):
        print(item,' ', obj, ' ', model_name, 'Fig_size: ', fig_size )
        print('Runing ', g, ' time..............')
        


        x_train_o = torch.from_numpy(trainX)
        x_test = torch.from_numpy(testX)
        
        y_train_p_o = torch.from_numpy(trainPlantY)
        y_test_p = torch.from_numpy(testPlantY)        
        y_train_d_o = torch.from_numpy(trainDiseaseY)
        y_test_d = torch.from_numpy(testDiseaseY)
        

        
        x_train, x_val, y_train_p, y_val_p, y_train_d, y_val_d = train_test_split(x_train_o,
                                                                                  y_train_p_o, y_train_d_o,
                                                                                  test_size=0.1,random_state=SEED)
        print("Number of sample for train: %d, val: %d,  test: %d"%(x_train.shape[0],x_val.shape[0],x_test.shape[0]))
    
    
        train_dataset = TensorDataset(x_train, y_train_p,y_train_d)  # train
        train_loader = DataLoader(dataset=train_dataset, batch_size=bat_si, shuffle=False,num_workers=0)
        train_data_size = len(train_dataset)
        
        val_dataset = TensorDataset(x_val, y_val_p, y_val_d)  # validation
        val_loader = DataLoader(dataset=val_dataset, batch_size=bat_si, shuffle=False,num_workers=0)
        val_data_size = len(val_dataset)
        
        test_dataset = TensorDataset(x_test, y_test_p, y_test_d) #test
        test_loader = DataLoader(dataset=test_dataset, batch_size=bat_si,num_workers=0)
        test_data_size = len(test_dataset)
        
        print('\n'+item+'_'+obj+' | ', model_name, '  fig_size: ', fig_size,'*', fig_size, 'start!\n' )

        # data_class = [numPlants, numDis]
        # data_class = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]
        # model = ResNetUNet2_2(fig_size, numPlants, numDis,dim_latent=16,n_fc=8)   
        model = ResNetUNet2_2_no_adain(fig_size, numPlants, numDis,dim_latent=16,n_fc=8)   




        optimizer = torch.optim.Adamax(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        model = model.to(device)
        criterion = criterion.to(device)

        dfhistory = pd.DataFrame(columns = ["epoch","loss","acc_p","acc_d","val_loss","val_acc_p","val_acc_d"]) 

        best_valid_loss = float('inf')
        va = 0
        for epoch in range(epo+1):
        
            start_time = time.monotonic()
        
            train_loss, train_acc_p, train_acc_d = train(model, train_loader, optimizer, criterion, device,latent_z)
            valid_loss, valid_acc_p, valid_acc_d = evaluate(model, val_loader, criterion, device,latent_z)

            info = (epoch, train_loss, train_acc_p, train_acc_d, 
                    valid_loss, valid_acc_p, valid_acc_d)
            dfhistory.loc[epoch] = info        
            if valid_loss < best_valid_loss:
                va = 0
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_1_'+str(fig_size)+'_d_'+DEVICES+'.pt')
                print('\nThe best model has been saved, best val_loss: ', round(best_valid_loss,5),', best epoch:',best_epoch,' \n')
            else:
                va += 1
                print('\nval_loss: not smaller | val step:', va)
                print('\ncurrent val_loss: ', round(valid_loss,5),' | best_val_loss: ', round(best_valid_loss,5),'\n')
                if va == early_stop:
                    break
            end_time = time.monotonic()
        
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc_p: {train_acc_p*100:.2f}% | Train Acc_d: {train_acc_d*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc_p: {valid_acc_p*100:.2f}% | Val. Acc_d: {valid_acc_d*100:.2f}%')

        model.load_state_dict(torch.load(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_1_'+str(fig_size)+'_d_'+DEVICES+'.pt'))
        x_test, y_plant_test, y_disease_test, y_plant_pred, y_disease_pred = get_predictions(model, test_loader, device)


        accuracy_plant = accuracy_score(y_plant_test, y_plant_pred)
        weighted_plant = f1_score(y_plant_test, y_plant_pred, average='weighted')
        
        accuracy_disease = accuracy_score(y_disease_test, y_disease_pred)
        weighted_disease = f1_score(y_disease_test, y_disease_pred, average='weighted')

        y_total_test_t = []
        z = 0
        for l in y_plant_test:   
            p_t = p_type[y_plant_test[z]]
            d_t = d_type[y_disease_test[z]]
            y_total_test_t.append((p_t+' '+d_t))
            z += 1
        
        y_total_pred_t = []
        x = 0
        for l in y_plant_pred:
            p_t = p_type[y_plant_pred[x]]
            d_t = d_type[y_disease_pred[x]]
            y_total_pred_t.append((p_t+' '+d_t))
            x += 1
        
        accuracy_total = accuracy_score(y_total_test_t, y_total_pred_t)
        weighted_total = f1_score(y_total_test_t, y_total_pred_t, average='weighted')        
        

        print( model_name+' | '+ item + '_'+ obj +' SKlearn Accuracy_total' ,accuracy_total,' SKlearn F1_score_total' ,weighted_total)#'\n',
        
        
        if saveornot == 'save':
            path_save = dir_save +'model_save/'+ model_name+'/'+obj+'_' + plus+'_No_'+str(g)+'/model_'+ str(round(accuracy_total,5))+'fig_size_' + str(fig_size)
            if not os.path.exists(path_save):
                print('Model & Result do not exist, make dir.')
                os.makedirs(path_save)
        
        plt.figure(1)
        acc_p = dfhistory['acc_p']
        acc_d = dfhistory['acc_d']
        val_acc_p = dfhistory['val_acc_p']
        val_acc_d = dfhistory['val_acc_d']
        plt.plot(acc_p, label='Plant_acc') #, 'bo--')
        plt.plot(acc_d, label='Disease_acc') #,, 'bo--')
        plt.plot(val_acc_p, label='Plant_val_acc') #, 'ro-')
        plt.plot(val_acc_d, label='Disease_val_acc') #, 'ro-')
        plt.title('Training and validation accuracy')
        plt.xlabel("Epoch")
        plt.ylabel('Accuracy')
        plt.legend(loc=0)
        if saveornot == 'save':
            plt.savefig(path_save+'/Acc&Val_acc.png',dpi=1000)
        plt.show()
        
        
        plt.figure(2)
        # loss_df = dfhistory['loss']
        # val_loss_df = dfhistory['val_loss']
        plt.plot(dfhistory['loss'],label='Loss')
        plt.plot(dfhistory['val_loss'],label='Val_loss')
        # plt.title(str(model_name)+"'s loss")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc=0)
        if saveornot == 'save':
            plt.savefig(path_save+'/Loss&Val_loss.png',dpi=1000)
        plt.show()
        
        
        pd.DataFrame(y_plant_test).to_csv(path_save +'/y_plant_test_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_plant_pred).to_csv(path_save +'/y_plant_pred_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_disease_test).to_csv(path_save +'/y_disease_test_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_disease_pred).to_csv(path_save +'/y_disease_pred_'+item + '_'+obj+'_'+model_name+'_.csv')
        
        pd.DataFrame(y_total_test_t).to_csv(path_save +'/y_total_test_t_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_total_pred_t).to_csv(path_save +'/y_total_pred_t_'+item + '_'+obj+'_'+model_name+'_.csv')   
        
        with open(dir_save+item +'_'+ obj + '_'+model_name+'_hist.txt', 'a') as h:
            h.write(item +'_'+ obj + '_' + str(model_name) + '  fig_size: '+ str(fig_size)+ '  batch_size: '+str(bat_si)+
                    '  best_epo: '+str(best_epoch)+'  No_'+str(g)+'  Accuracy_plant: '+str(accuracy_plant)+ '  F1_score_plant: '+str(weighted_plant)+
                    '  Accuracy_disease: '+str(accuracy_disease)+ '  F1_score_disease: '+str(weighted_disease)+
                    '  Accuracy_total: '+str(accuracy_total)+ '  F1_score_total: '+str(weighted_total)+'\n')    

        print( 'Accuracy_plant' ,accuracy_plant)#'\n',
        print( 'F1_score_plant' ,weighted_plant)#'\n',
        print( 'Accuracy_disease' ,accuracy_disease)#'\n',
        print( 'F1_score_disease' ,weighted_disease)#'\n',            
        print( 'No.'+ str(g)+' total_Accuracy' ,accuracy_total)
        print( 'No.'+ str(g)+' total_F1_score' ,weighted_total)        
        
        ten_accuracy.append(accuracy_total)
        ten_weighted.append(weighted_total)
        
        ten_accuracy_plant.append(accuracy_plant)
        ten_weighted_plant.append(weighted_plant)
        ten_accuracy_disease.append(accuracy_disease)
        ten_weighted_disease.append(weighted_disease)        
        torch.save(model.state_dict(), path_save+'/'+item + '_'+obj+'_'+model_name+'_pytorch.pt')
        

    mean_acc = np.mean(ten_accuracy)
    mean_weighted = np.mean(ten_weighted)
    
    mean_accuracy_plant = np.mean(ten_accuracy_plant)
    mean_weighted_plant = np.mean(ten_weighted_plant)
    mean_accuracy_disease = np.mean(ten_accuracy_disease)
    mean_weighted_disease = np.mean(ten_weighted_disease)
    
    # import scipy.stats as st
    std_acc = np.std(ten_accuracy, axis=0)
    std_f1 = np.std(ten_weighted, axis=0)
    
    std_acc_plant = np.std(ten_accuracy_plant, axis=0)
    std_f1_plant = np.std(ten_weighted_plant, axis=0)
    std_acc_disease = np.std(ten_accuracy_disease, axis=0)
    std_f1_disease = np.std(ten_weighted_disease, axis=0)

    print('\n',model_name + ' | '+item+'_'+ obj+ plus  + " | Fig_size: ", fig_size, '*', fig_size, " Done!")

    print( 'Mean_Accuracy_plant' ,mean_accuracy_plant, ' std_acc_plant: ', str(std_acc_plant))
    print( 'Mean_F1_score_plant' ,mean_weighted_plant, ' std_f1_plant: ', str(std_f1_plant))
    print( 'Mean_Accuracy_disease' ,mean_accuracy_disease, ' std_acc_dis: ', str(std_acc_disease))
    print( 'Mean_F1_score_disease' ,mean_weighted_disease, ' std_f1_dis: ', str(std_f1_disease))
    print( 'Mean_Accuracy_total' ,mean_acc, ' std_acc: ', str(std_acc))
    print( 'Mean_F1_score_total' ,mean_weighted, ' std_f1: ', str(std_f1))    

    with open(dir_save+obj+'_'+ plus +'_hist_'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
        h.write(item+'_'+obj+'_' + plus+' 10 times | ' + str(model_name) +'  Mean_Accuracy: '+str(mean_acc)+' std_acc_plant: '+ str(std_acc)+ '  Mean_F1_score: '+str(mean_weighted)+' std_f1: '+str(std_f1)+'\n'+'\n')    

    with open(dir_save+obj+'_'+ plus +'_hist_plant&dis'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
        h.write(item+'_'+obj+'_'+ plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_plant: '+str(mean_accuracy_plant)+' std_acc_plant: '+ str(std_acc_plant)+ '  Mean_F1_score_plant: '+str(mean_weighted_plant)+' std_f1_plant: '+str(std_f1_plant)+'\n'+'\n')    
        h.write(item+'_'+obj+'_'+ plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_disease: '+str(mean_accuracy_disease)+' std_acc_disease: '+ str(std_acc_disease)+ '  Mean_F1_score_disease: '+str(mean_weighted_disease)+' std_f1_disease: '+str(std_f1_disease)+'\n'+'\n')    
        
    np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ plus +'_' + str(model_name)+'_Accuracy_10_times.txt', ten_accuracy,fmt='%s',delimiter=',')
    np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ plus +'_' + str(model_name)+'_Weighted_10_times.txt', ten_weighted,fmt='%s',delimiter=',')
        
    
print('All Done!')    



