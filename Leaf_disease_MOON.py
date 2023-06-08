# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:40:36 2022

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
# from ipywidgets import IntProgress
from tqdm import tqdm

early_stop = 50

DEVICES = '0'
device = torch.device('cuda:'+DEVICES+'' if torch.cuda.is_available() else 'cpu')
# device_ids = [0, 1] 
# device=device_ids[0]




import load_data
torch.cuda.empty_cache()

SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



####################################
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    y = y.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# from tqdm.notebook import trange, tqdm
def train(model, iterator, optimizer, criterion, device):

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
        y_pred_p,y_pred_d = model(x.float())
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


def evaluate(model, iterator, criterion, device):

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
            
            y_pred_p,y_pred_d  = model(x.float())

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
            y_pr_p, y_pr_d  = model(x.float())

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
 This code of the MOON was modified from:
#  https://github.com/QinbinLi/MOON/blob/main/model.py
Their paper:
@inproceedings{li2021model,
      title={Model-Contrastive Federated Learning}, 
      author={Qinbin Li and Bingsheng He and Dawn Song},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2021},
}


'''

# base_model = 'simple-cnn'
base_model = "resnet50-cifar10" 
# base_model = "resnet18-cifar10"

class MLP_header(nn.Module):
    def __init__(self,out_dim):
        super(MLP_header, self).__init__()
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        #projection
        # self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # x = x.view(-1, 28*28)
        print('here: ', x.shape)
        x = self.fc1(x)
        print('look: ',x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        # self.fc1 = nn.Linear(7424, hidden_dims[0])
        self.fc1 = nn.Linear(13456, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = x.view(-1, 128)
        # print(x.shape)
        x = torch.flatten(x, 1) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCifar10(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar10, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def ResNet18_cifar10(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet50_cifar10(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(Bottleneck, [3, 4, 6, 3], **kwargs)

class ModelFedCon(nn.Module):

    # def __init__(self, base_model, out_dim, n_classes, net_configs=None):
    def __init__(self, base_model, out_dim, numPlants, numDis, net_configs=None):        
        super(ModelFedCon, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header(out_dim)
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=out_dim, hidden_dims=[120, 84], output_dim=numDis)
            num_ftrs = 84
        # elif base_model == 'simple-cnn-mnist':
        #     self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
        #     num_ftrs = 84

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        
        self.fc_1 =nn.Linear(out_dim, out_dim)     
        # last layer
        # self.l3 = nn.Linear(out_dim, n_classes)
        self.fc_p =nn.Linear(out_dim, numPlants)
        self.fc_d =nn.Linear(out_dim, numDis)   
        

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        #print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        y = self.l2(x)

        # y = self.l3(x)
        
        y = torch.flatten(y, 1) 
        y= F.relu(self.fc_1(y))
        
        out_p = F.log_softmax(self.fc_p(y), -1)
        out_d = F.log_softmax(self.fc_d(y), -1)
        
        return out_p, out_d



#########################################################
def MOON(item, obj,dataset_dir,save_path,saveornot, fig_size = 256,  bat_si = 16,INIT_LR = 0.001, epo = 10000, times = 10, op_z = "Adamax"):

    early_stop = 50

    model_name = 'MOON'
    plus = "_"
    save_dir= str(save_path) + str(item)+'_'+str(obj) +'/'
    
    if item == "PlantDoc_original":
        trainX, trainDiseaseY, trainPlantY, categ, dir_save, p_type, d_type, testX, disease_y, plant_y = load_data.load_data(dataset_dir,save_dir, model_name, item, obj, fig_size)
        trainX=np.array(trainX).reshape(-1,3, fig_size,fig_size)
        diseaseLB = LabelBinarizer()
        PlantLB = LabelBinarizer()
        trainDiseaseY = diseaseLB.fit_transform(trainDiseaseY)
        trainPlantY = PlantLB.fit_transform(trainPlantY)
        
        numDis=len(diseaseLB.classes_)
        numPlants=len(PlantLB.classes_) 
        
        testX=np.array(testX).reshape(-1,3, fig_size,fig_size)
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
        train_loader = DataLoader(dataset=train_dataset, batch_size=bat_si, shuffle=True,num_workers=0)
        train_data_size = len(train_dataset)
        
        val_dataset = TensorDataset(x_val, y_val_p, y_val_d)  # validation
        val_loader = DataLoader(dataset=val_dataset, batch_size=bat_si, shuffle=True,num_workers=0)
        val_data_size = len(val_dataset)
        
        test_dataset = TensorDataset(x_test, y_test_p, y_test_d) #test
        test_loader = DataLoader(dataset=test_dataset, batch_size=bat_si,num_workers=0, shuffle=False,)
        test_data_size = len(test_dataset)
        
        print('\n'+item+'_'+obj+' | ', model_name, '  fig_size: ', fig_size,'*', fig_size, 'start!\n' )


        model = ModelFedCon(base_model, fig_size, numPlants, numDis, net_configs=None)   

        
        optimizer = torch.optim.Adamax(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # mode=torch.nn.DataParallel(model,device_ids=device_ids)
        # mode=mode.cuda(device)

        model = model.to(device)
        criterion = criterion.to(device)

        dfhistory = pd.DataFrame(columns = ["epoch","loss","acc_p","acc_d","val_loss","val_acc_p","val_acc_d"]) 

        best_valid_loss = float('inf')
        va = 0
        for epoch in range(epo+1):
        
            start_time = time.monotonic()
        
            train_loss, train_acc_p, train_acc_d = train(model, train_loader, optimizer, criterion, device)
            valid_loss, valid_acc_p, valid_acc_d = evaluate(model, val_loader, criterion, device)

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
            path_save = dir_save +'model_save/'+ model_name+'/'+obj+'_'+'_' + plus+'_No_'+str(g)+'/model_'+ str(round(accuracy_total,5))+'fig_size_' + str(fig_size)
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
            plt.savefig(path_save+'/Acc&Val_acc.png',dpi=200)
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
            plt.savefig(path_save+'/Loss&Val_loss.png',dpi=200)
        plt.show()
        
        if saveornot == 'save':        
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

    print('\n',model_name + ' | '+item+'_'+ obj+'_'+plus  + " | Fig_size: ", fig_size, '*', fig_size, " Done!")

    print( 'Mean_Accuracy_plant' ,mean_accuracy_plant, ' std_acc_plant: ', str(std_acc_plant))
    print( 'Mean_F1_score_plant' ,mean_weighted_plant, ' std_f1_plant: ', str(std_f1_plant))
    print( 'Mean_Accuracy_disease' ,mean_accuracy_disease, ' std_acc_dis: ', str(std_acc_disease))
    print( 'Mean_F1_score_disease' ,mean_weighted_disease, ' std_f1_dis: ', str(std_f1_disease))
    print( 'Mean_Accuracy_total' ,mean_acc, ' std_acc: ', str(std_acc))
    print( 'Mean_F1_score_total' ,mean_weighted, ' std_f1: ', str(std_f1))    

    with open(dir_save+obj+'_'+plus +'_hist_'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
        h.write(item+'_'+obj+'_'+plus+' 10 times | ' + str(model_name) +'  Mean_Accuracy: '+str(mean_acc)+' std_acc_plant: '+ str(std_acc)+ '  Mean_F1_score: '+str(mean_weighted)+' std_f1: '+str(std_f1)+'\n'+'\n')    

    with open(dir_save+obj+'_'+plus +'_hist_plant&dis'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
        h.write(item+'_'+obj+'_'+plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_plant: '+str(mean_accuracy_plant)+' std_acc_plant: '+ str(std_acc_plant)+ '  Mean_F1_score_plant: '+str(mean_weighted_plant)+' std_f1_plant: '+str(std_f1_plant)+'\n'+'\n')    
        h.write(item+'_'+obj+'_'+plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_disease: '+str(mean_accuracy_disease)+' std_acc_disease: '+ str(std_acc_disease)+ '  Mean_F1_score_disease: '+str(mean_weighted_disease)+' std_f1_disease: '+str(std_f1_disease)+'\n'+'\n')    
        
    np.savetxt(dir_save +'/'+item+'_'+obj+'_'+plus +'_' + str(model_name)+'_Accuracy_10_times.txt', ten_accuracy,fmt='%s',delimiter=',')
    np.savetxt(dir_save +'/'+item+'_'+obj+'_'+plus +'_' + str(model_name)+'_Weighted_10_times.txt', ten_weighted,fmt='%s',delimiter=',')
        
    
print('All Done!')    



