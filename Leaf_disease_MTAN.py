# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:04:39 2022

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

        loss_1 = criterion(y_pred_p, y_p.float())
        loss_2 = criterion(y_pred_d, y_d.float())
        loss = loss_1+loss_2

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
 This code of the MTAN was modified from:
  https://github.com/lorenmt/mtan/blob/master/visual_decathlon/model_wrn_mtan.py
Their paper:
@inproceedings{liu2019end,
  title={End-to-End Multi-task Learning with Attention},
  author={Liu, Shikun and Johns, Edward and Davison, Andrew J},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1871--1880},
  year={2019}
}


'''
import torch.nn.init as init
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet(nn.Module):
    def __init__(self,fig_size, numPlants, numDis, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)

        self.linear = nn.ModuleList([nn.Sequential(
            nn.Linear(filter[3], num_classes[0]),
            nn.Softmax(dim=1))])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        for j in range(len(num_classes)):
            if j < (len(num_classes)-1):
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.linear.append(nn.Sequential(nn.Linear(filter[3], num_classes[j + 1]),
                                                 nn.Softmax(dim=1)))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
        # self.max_2 = nn.MaxPool2d(8)             
        if fig_size == 256:
            self.fc_1 = nn.Linear(64*8*8, 256) # fig_size == 256:
        elif fig_size == 128:
            self.fc_1 =nn.Linear(32*4*8, 128) #  fig_size == 128:
        # self.relu_6 =nn.ReLU(inplace=False)                              
                
        self.fc_p =nn.Linear(fig_size, numPlants)
        self.fc_d =nn.Linear(fig_size, numDis)                   
    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, k=1):
        g_encoder = [0] * 4

        atten_encoder = [0] * 10
        for i in range(10):
            atten_encoder[i] = [0] * 4
        for i in range(10):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3

        # shared encoder
        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))

        # apply attention modules
        for j in range(4):
            if j == 0:
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](torch.cat((g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                if j < 3:
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)
        # print("LOOK:  ",  atten_encoder[k][-1][-1])
        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8)
        pred = pred.view(pred.size(0), -1)

        # out = self.linear[k](pred)
        # return out


        x = torch.flatten(pred, 1) # flatten all dimensions except batch          
        x = F.relu(self.fc_1(x))
        out_p = F.log_softmax(self.fc_p(x), -1)
        out_d = F.log_softmax(self.fc_d(x), -1)
        # x = self.fc3(x)
        return out_p,out_d



#############################################    



def MTAN(item, obj,dataset_dir,save_path,saveornot, fig_size = 256,  bat_si = 16,INIT_LR = 0.001, epo = 10000, times = 10, op_z = "Adamax"):

    early_stop = 50

    model_name = 'MTAN'
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
        

        x_train = torch.from_numpy(trainX)
        x_test = torch.from_numpy(testX)
        
        y_train_p = torch.from_numpy(trainPlantY)
        y_test_p = torch.from_numpy(testPlantY)        
        y_train_d = torch.from_numpy(trainDiseaseY)
        y_test_d = torch.from_numpy(testDiseaseY)
        
    
        
        x_train, x_val, y_train_p, y_val_p, y_train_d, y_val_d = train_test_split(x_train,y_train_p, y_train_d, test_size=0.1,random_state=SEED)
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
    
        data_class = [numPlants, numDis]
        # data_class = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]
        model = WideResNet(fig_size, numPlants, numDis, depth=28, widen_factor = 4, num_classes=data_class )   
    
        optimizer = torch.optim.Adamax(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
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
    

    std_acc = np.std(ten_accuracy, axis=0)
    std_f1 = np.std(ten_weighted, axis=0)
    
    std_acc_plant = np.std(ten_accuracy_plant, axis=0)
    std_f1_plant = np.std(ten_weighted_plant, axis=0)
    std_acc_disease = np.std(ten_accuracy_disease, axis=0)
    std_f1_disease = np.std(ten_weighted_disease, axis=0)
    
    print('\n',model_name + ' | '+item+'_'+ obj+'_'+ plus  + " | Fig_size: ", fig_size, '*', fig_size, " Done!")
    
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
    
    
    
