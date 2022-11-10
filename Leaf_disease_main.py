# -*- coding: utf-8 -*-

import tensorflow as tf
print("tensorflow version: ", tf.__version__)
from tensorflow.keras import models,callbacks
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import os,load_data,train_models   
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
import argparse,sys
# multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# ########################################################


def main(item, obj, model_name,dataset_dir,save_path,saveornot, fig_size = 256,  bat_si = 16,INIT_LR = 0.001, epo = 10000, times = 10, op_z = "Adamax", TF_weights=None,balance_weight = [0.1,0.1,0.4,0.5]):
    plus = "_"
    save_dir= str(save_path) + str(item)+'_'+str(obj) +'/'
    
    w_p = balance_weight[2] 
    w_d = balance_weight[3] 
    w_p_t = balance_weight[0] 
    w_d_t = balance_weight[1] 
    
    
    if obj == "multi_label":
        x = []
        y = []
        PlantLabels = []
        diseaseLabels = []
        if item == "PlantDoc_original":
            dataset, categ, dir_save, categories, p_type, d_type, test_set = load_data.load_data(dataset_dir,save_dir, model_name, item, obj, fig_size)
    
            x_train = []
            y_train = []
            for features, label in dataset:
                x_train.append(features)
                y_train.append(label)
            x_train=np.array(x_train).reshape(-1,fig_size,fig_size,3)
            
            x_test = []
            y_total_test = []
            for features, label in test_set:
                x_test.append(features)
                y_total_test.append(label)
            x_test=np.array(x_test).reshape(-1,fig_size,fig_size,3)
    
    
        else:
            dataset, categ, dir_save,categories, p_type, d_type = load_data.load_data(dataset_dir,save_dir, model_name, item, obj, fig_size)
    
            for features, label in dataset:
                x.append(features)
                y.append(label)
    
            x=np.array(x).reshape(-1,fig_size,fig_size,3)
    
            split = train_test_split(x, y, test_size=0.2, random_state = 50)
            (x_train, x_test, y_train, y_total_test) = split
            
    else:
    
        if item == "PlantDoc_original":
            trainX, trainDiseaseY, trainPlantY,categ, dir_save, p_type, d_type, testX, disease_y, plant_y = load_data.load_data(dataset_dir,save_dir, model_name, item, obj, fig_size)
            trainX=np.array(trainX).reshape(-1,fig_size,fig_size,3)
            diseaseLB = LabelBinarizer()
            PlantLB = LabelBinarizer()
            trainDiseaseY = diseaseLB.fit_transform(trainDiseaseY)
            trainPlantY = PlantLB.fit_transform(trainPlantY)
            
            numDis=len(diseaseLB.classes_)
            numPlants=len(PlantLB.classes_) 
            
            testX=np.array(testX).reshape(-1,fig_size,fig_size,3)
            disease_y = np.array(disease_y)
            plant_y = np.array(plant_y)
        
            testDiseaseY = diseaseLB.fit_transform(disease_y)
            testPlantY = PlantLB.fit_transform(plant_y)
            
            
        else:
            data, diseaseLabels, PlantLabels, categ, dir_save, p_type, d_type  = load_data.load_data(dataset_dir,save_dir, model_name, item, obj, fig_size)
            
            data=np.array(data).reshape(-1,fig_size,fig_size,3)
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
    
        
    
    
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.003, patience=50, verbose=1, mode='auto') 
    mc = callbacks.ModelCheckpoint(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    
    
    ten_accuracy = []
    ten_weighted = []
    ten_accuracy_plant = []
    ten_weighted_plant = []
    ten_accuracy_disease = []
    ten_weighted_disease = []
    
    ten_accuracy_t = []
    ten_weighted_t = []
    ten_accuracy_plant_t = []
    ten_weighted_plant_t = []
    ten_accuracy_disease_t = []
    ten_weighted_disease_t = []  
    
    inputShape = (fig_size, fig_size, 3)
    input_layer = Input(shape=inputShape)
    
    for g in range(0,times):
        print(item,' ', obj, ' ', model_name )
        print('Runing ', g, ' time..............')
        
    
    #################################################################################
    
        K.clear_session()
        if obj == "multi_label":
            numPlants = 0
            numDis = 0
            model = train_models.multi_output_model(model_name,input_layer,numPlants,numDis,categ, item, w_p, w_d, w_p_t,w_d_t, TF_weights,obj,op_z,INIT_LR)
            
            y_train_cat=tf.one_hot(y_train,categ)  #Tensorflow 2
            # y_test_cat=tf.one_hot(y_total_test,categ)
            
            #fit the model i.e. training the model and batch size can be varies
            H=model.fit(x_train,y_train_cat,batch_size=bat_si, #16,
                      epochs=epo,verbose=1,
                      validation_split=0.1,
                       shuffle=True,
                       callbacks=[es,mc]
                      )
        
            # ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'.h5')
            ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5')
            
    
            predict_x=ModelCheck_model.predict(x_test)            # tf >= 2.6
            y_total_pred=np.argmax(predict_x,axis=1)
        
        
    
            
    #y_plant_test, y_plant_pred 
        
            y_disease_test=[]
            y_plant_test=[]
            z = 0
            for n in y_total_test:
                # print(categories[n])
                if item ==  "plant_village":
                    y_plant_t = p_type.index(categories[n][:categories[n].index("___")])
                    y_disease_t = d_type.index((categories[n][categories[n].index("___")+3:]))
                elif item ==  "plant_leaves":
                    y_plant_t = p_type.index(categories[n][:categories[n].index("_")])
                    y_disease_t = (categories[n][categories[n].index("_")+1:])
                    if y_disease_t == 'healthy':
                        pass
                    else:
                        y_disease_t = categories[n]
                    y_disease_t = d_type.index(y_disease_t)
                elif item ==  "PlantDoc" or item == "PlantDoc_original":
                    y_plant_t = p_type.index(categories[n][:categories[n].index(" ")])
                    y_disease_t = (categories[n][categories[n].index(" ")+1:])
                    if y_disease_t == 'leaf':
                        y_disease_t = 'healthy'
                        # print(n_p_type,'   ','healthy')
                    elif y_disease_t == 'Early blight leaf':
                        y_disease_t = 'leaf early blight'
                        # print(n_p_type,'   ','leaf early blight')
                    else:
                        pass
                    y_disease_t = d_type.index(y_disease_t)
                    
                else:
                    print('error!')
                
                y_plant_test.append(y_plant_t)
                y_disease_test.append(y_disease_t)
                z += 1
        
                
            y_disease_pred=[]
            y_plant_pred=[]
            x = 0
            for l in y_total_pred:
                if item ==  "plant_village":          
                    y_plant_p=p_type.index(categories[l][:categories[l].index("___")])
                    y_disease_p = d_type.index((categories[l][categories[l].index("___")+3:]))
                elif item ==  "plant_leaves":
                    y_plant_p=p_type.index(categories[l][:categories[l].index("_")])
                    y_disease_p = (categories[l][categories[l].index("_")+1:])
                    if y_disease_p == 'healthy':
                        pass
                    else:
                        y_disease_p = categories[l]
                    y_disease_p = d_type.index(y_disease_p)                
                elif item ==  "PlantDoc" or item == "PlantDoc_original":
                    y_plant_p=p_type.index(categories[l][:categories[l].index(" ")])
                    y_disease_p = (categories[l][categories[l].index(" ")+1:])
                    if y_disease_p == 'leaf':
                        y_disease_p = 'healthy'
                        # print(n_p_type,'   ','healthy')
                    elif y_disease_p == 'Early blight leaf':
                        y_disease_p = 'leaf early blight'
                        # print(n_p_type,'   ','leaf early blight')
                    else:
                        pass
                    y_disease_p = d_type.index(y_disease_p)
        
                else:
                    print('error!')                
                y_plant_pred.append(y_plant_p)
                y_disease_pred.append(y_disease_p)
                x += 1
    
    
        else:
            if obj == "multi_model":
                model = train_models.multi_model(model_name,input_layer,numPlants,numDis,item, TF_weights,op_z,INIT_LR)
                H = model.fit(x=trainX,
                              y={"disease_output": trainDiseaseY, "plant_output": trainPlantY},
                              batch_size=bat_si,
                              validation_split=0.1,
                              shuffle=True,
                              epochs=epo,
                              verbose=1,
                              callbacks=[es,mc]
                              )
                ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5')
                (disease_y_pred, plant_y_pred)=ModelCheck_model.predict(testX)
            elif obj == "multi_output":
                model = train_models.multi_output_model(model_name,input_layer,numPlants,numDis,categ, item, w_p, w_d, w_p_t,w_d_t, TF_weights,obj,op_z,INIT_LR)
                
                H = model.fit(x=trainX,
                              y={"disease_output": trainDiseaseY, "plant_output": trainPlantY},batch_size=bat_si,
                              validation_split=0.1,
                              shuffle=True,
                              epochs=epo,
                              verbose=1,
                              callbacks=[es,mc]
                              )
        
        
                ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5')
                (disease_y_pred, plant_y_pred)=ModelCheck_model.predict(testX)
                
            elif obj == "new_model":
                model = train_models.multi_output_model(model_name,input_layer,numPlants,numDis,categ, item, w_p, w_d, w_p_t,w_d_t, TF_weights,obj,op_z,INIT_LR)
                if item == "PlantDoc" or item == "PlantDoc_original":    
                    callback=[es]
                else:
                    callback=[es,mc]
                    pass
                H = model.fit(x=trainX,
                              y=[trainPlantY,trainDiseaseY,trainPlantY,trainDiseaseY],batch_size=bat_si,
                              validation_split=0.1,
                              shuffle=True,
                              epochs=epo,
                              verbose=1,
                              callbacks=callback
                              )
        
                if item == "PlantDoc" or item == "PlantDoc_original": 
                    (plant_y_pred,disease_y_pred,plant_y_pred_t,disease_y_pred_t)=model.predict(testX)
                else:
                    ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5')
                    (plant_y_pred,disease_y_pred,plant_y_pred_t,disease_y_pred_t)=ModelCheck_model.predict(testX)  
            
            elif obj == "cross_stitch":
                model, CrossStitchBlock = train_models.cross_stitch(model_name,input_layer,numPlants,numDis,item,op_z,INIT_LR)
    
                H = model.fit(x=trainX,
                              y=[trainPlantY,trainDiseaseY],batch_size=bat_si,
                              validation_split=0.1,
                              shuffle=True,
                              epochs=epo,
                              verbose=1,
                              callbacks=[es,mc]
                              )
    
                ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5', custom_objects={'CrossStitchBlock': CrossStitchBlock})
                
                
                (plant_y_pred,disease_y_pred)=ModelCheck_model.predict(testX)
    
            y_disease_pred=np.argmax(disease_y_pred,axis=1)
            y_plant_pred=np.argmax(plant_y_pred,axis=1)
            
            y_disease_test = np.argmax(testDiseaseY,axis=1)
            y_plant_test = np.argmax(testPlantY,axis=1)
     
        accuracy_plant = accuracy_score(y_plant_test, y_plant_pred)
        weighted_plant = f1_score(y_plant_test, y_plant_pred, average='weighted')
        # cm_plant = confusion_matrix(y_plant_test, y_plant_pred)#, labels=categories)
        
        accuracy_disease = accuracy_score(y_disease_test, y_disease_pred)
        weighted_disease = f1_score(y_disease_test, y_disease_pred, average='weighted')
        # cm_disease = confusion_matrix(y_disease_test, y_disease_pred)#, labels=categories)
        if obj == "new_model" or obj == "multi_model" or obj == "multi_output" or obj == "cross_stitch":
            y_total_test = []
            z = 0
            for l in y_plant_test:   
                p_t = p_type[y_plant_test[z]]
                d_t = d_type[y_disease_test[z]]
                y_total_test.append((p_t+' '+d_t))
                z += 1
            
            y_total_pred = []
            x = 0
            for l in y_plant_pred:
                p_t = p_type[y_plant_pred[x]]
                d_t = d_type[y_disease_pred[x]]
                y_total_pred.append((p_t+' '+d_t))
                x += 1
        
        accuracy_total = accuracy_score(y_total_test, y_total_pred)
        weighted_total = f1_score(y_total_test, y_total_pred, average='weighted')        
    
    
    ##########################################################################################################
        if obj == "new_model":
            y_disease_pred_t=np.argmax(disease_y_pred_t,axis=1)
            y_plant_pred_t=np.argmax(plant_y_pred_t,axis=1)
            
        
            
            accuracy_plant_t = accuracy_score(y_plant_test, y_plant_pred_t)
            weighted_plant_t = f1_score(y_plant_test, y_plant_pred_t, average='weighted')
            # cm_plant_t = confusion_matrix(y_plant_test, y_plant_pred_t)#, labels=categories)
            
            accuracy_disease_t = accuracy_score(y_disease_test, y_disease_pred_t)
            weighted_disease_t = f1_score(y_disease_test, y_disease_pred_t, average='weighted')
            # cm_disease_t = confusion_matrix(y_disease_test, y_disease_pred_t)#, labels=categories)
        
        
            
            y_total_pred_t = []
            x = 0
            for l in y_plant_pred:
                p_t_t = p_type[y_plant_pred_t[x]]
                d_t_t = d_type[y_disease_pred_t[x]]
                y_total_pred_t.append((p_t_t+' '+d_t_t))
                x += 1
            
            accuracy_total_t = accuracy_score(y_total_test, y_total_pred_t)
            weighted_total_t = f1_score(y_total_test, y_total_pred_t, average='weighted')     
        
        ##########################################################################################
    
    
        # cm_plant = confusion_matrix(y_plant_test, y_plant_pred)#, labels=categories)
        # print( model_name+' | '+ item + '_'+ obj +' SKlearn Accuracy_total' ,accuracy_total,' SKlearn F1_score_total' ,weighted_total)#'\n',
        # print( model_name+' | '+ item + '_'+ obj +' SKlearn Accuracy_total_t' ,accuracy_total_t,' SKlearn F1_score_total_t' ,weighted_total_t)#'\n',
                
       
    
    
    
    
        model_history = H.history
        if saveornot == 'save':
            path_save = dir_save +'model_save/'+ model_name+'/'+obj+'_' + plus+'_No_'+str(g)+'/model_'+ str(round(weighted_total,5))+'fig_size_' + str(fig_size)
            if not os.path.exists(path_save):
                print('Model & Result do not exist, make dir.')
                os.makedirs(path_save)
        if obj == "multi_label":        
            plt.figure(1,dpi=200)
            plt.plot(model_history['accuracy'],label='accuracy')
            plt.plot(model_history['val_accuracy'],label='val_accuracy')
            plt.title(str(model_name)+"'s Accuracy"+ '| total_F1_score: ' + str(round(weighted_total,5))+'%')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(loc=0) #['Train', 'Test']
            # plt.savefig('accuracy')
            plt.tight_layout()
            if saveornot == 'save':
                plt.savefig(path_save+'/Acc&Val_acc.png',dpi=200)
            plt.show()
            
            plt.figure(2,dpi=200)
            plt.plot(model_history['loss'],label='loss')
            plt.plot(model_history['val_loss'],label='val_loss')
            plt.title(str(model_name)+"'s loss"+ '| total_F1_score: ' + str(round(weighted_total,5))+'%')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc=0)
            # plt.savefig('loss')
            plt.tight_layout()
            if saveornot == 'save':
                plt.savefig(path_save+'/Loss&Val_loss.png',dpi=200)
            # plt.show()
            
        else:        
            plt.figure(1,dpi=200)
            # plt.plot(H.history['accuracy'])
            plt.plot(H.history['disease_output_accuracy'],label='disease_output_accuracy')
            plt.plot(H.history['plant_output_accuracy'],label='plant_output_accuracy')
            plt.plot(H.history['val_disease_output_accuracy'],label='val_disease_output_accuracy')
            plt.plot(H.history['val_plant_output_accuracy'],label='val_plant_output_accuracy')
            plt.title(str(model_name)+"'s Accuracy"+ '| total_F1_score: ' + str(round(weighted_total,5))+'%')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(loc=0) #['Train', 'Test']
            # plt.savefig('accuracy')
            plt.tight_layout()
            if saveornot == 'save':
                plt.savefig(path_save+'/Acc&Val_acc.png',dpi=200)
            # plt.show()
            
            
            plt.figure(2,dpi=200)
            plt.plot(H.history['loss'],label='loss')
            plt.plot(H.history['val_loss'],label='val_loss')
            plt.plot(H.history['val_disease_output_loss'],label='val_disease_output_loss')
            plt.plot(H.history['val_plant_output_loss'],label='val_plant_output_loss')
            plt.title(str(model_name)+"'s loss"+ '| total_F1_score: ' + str(round(weighted_total,5))+'%')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc=0)
            # plt.savefig('loss')
            plt.tight_layout()
            if saveornot == 'save':
                plt.savefig(path_save+'/Loss&Val_loss.png',dpi=200)
            # plt.show()
    
     
    
        if saveornot == 'save':
            pd.DataFrame(y_plant_test).to_csv(path_save +'/y_plant_test_'+item + '_'+obj+'_'+model_name+'.csv')
            pd.DataFrame(y_plant_pred).to_csv(path_save +'/y_plant_pred_'+item + '_'+obj+'_'+model_name+'.csv')
            pd.DataFrame(y_disease_test).to_csv(path_save +'/y_disease_test_'+item + '_'+obj+'_'+model_name+'.csv')
            pd.DataFrame(y_disease_pred).to_csv(path_save +'/y_disease_pred_'+item + '_'+obj+'_'+model_name+'.csv')
            
            pd.DataFrame(y_total_test).to_csv(path_save +'/y_total_test_both_'+item + '_'+obj+'_'+model_name+'.csv', sep = ',' )
            pd.DataFrame(y_total_pred).to_csv(path_save +'/y_total_pred_both_'+item + '_'+obj+'_'+model_name+'.csv')   
            if obj == "new_model":
                pd.DataFrame(y_plant_pred_t).to_csv(path_save +'/y_plant_pred_t_'+item + '_'+obj+'_'+model_name+'.csv')
                pd.DataFrame(y_disease_pred_t).to_csv(path_save +'/y_disease_pred_t_'+item + '_'+obj+'_'+model_name+'_.csv')
                pd.DataFrame(y_total_pred_t).to_csv(path_save +'/y_total_pred_t_'+item + '_'+obj+'_'+model_name+'.csv')           
     
        
            # from tensorflow.keras.utils import plot_model
            # plot_model(model, to_file = path_save+'/'+item + '_'+obj+'_'+model_name+'.png',
            #            show_shapes=True,show_dtype=True,show_layer_names=True)
            
            # with open(path_save + '/model_summary.txt', 'w') as f:
            #     model.summary(print_fn=lambda x: f.write(x + '\n'))
            # model.save(path_save+'/'+item + '_'+obj+'_'+model_name+'_multi_tasks.h5')
        
            print('Saved to ', path_save)
    
            with open(dir_save+item +'_'+ obj + '_'+model_name+'_hist.txt', 'a') as h:
                h.write(item +'_'+ obj + '_' + str(model_name) + '  fig_size: '+ str(fig_size)+ '  batch_size: '+str(bat_si)+
                        '  epochs: '+str(epo)+'  Accuracy_plant: '+str(accuracy_plant)+ '  F1_score_plant: '+str(weighted_plant)+
                        '  Accuracy_disease: '+str(accuracy_disease)+ '  F1_score_disease: '+str(weighted_disease)+
                        '  Accuracy_total: '+str(accuracy_total)+ '  F1_score_total: '+str(weighted_total)+'\n')    
    
    
        print( 'No.'+ str(g))
        print( 'Accuracy_plant' ,accuracy_plant)#'\n',
        print( 'F1_score_plant' ,weighted_plant)#'\n',
        print( 'Accuracy_disease' ,accuracy_disease)#'\n',
        print( 'F1_score_disease' ,weighted_disease)#'\n',            
        print( 'total_Accuracy' ,accuracy_total)
        print( 'total_F1_score' ,weighted_total)        
        if obj == "new_model":
            print('--------------------------------------------------')
            print( 'Accuracy_plant_t' ,accuracy_plant_t)#'\n',
            print( 'F1_score_plant_t' ,weighted_plant_t)#'\n',
            print( 'Accuracy_disease_t' ,accuracy_disease_t)#'\n',
            print( 'F1_score_disease_t' ,weighted_disease_t)#'\n',            
            print( 'total_Accuracy_t' ,accuracy_total_t)
            print( 'total_F1_score_t' ,weighted_total_t)   
        
        ten_accuracy.append(accuracy_total)
        ten_weighted.append(weighted_total)
        ten_accuracy_plant.append(accuracy_plant)
        ten_weighted_plant.append(weighted_plant)
        ten_accuracy_disease.append(accuracy_disease)
        ten_weighted_disease.append(weighted_disease)
        if obj == "new_model":
            ten_accuracy_t.append(accuracy_total_t)
            ten_weighted_t.append(weighted_total_t)
            ten_accuracy_plant_t.append(accuracy_plant_t)
            ten_weighted_plant_t.append(weighted_plant_t)
            ten_accuracy_disease_t.append(accuracy_disease_t)
            ten_weighted_disease_t.append(weighted_disease_t)
    
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
    ###############################################################################################
    if obj == "new_model":
        mean_acc_t = np.mean(ten_accuracy_t)
        mean_weighted_t = np.mean(ten_weighted_t)
        
        mean_accuracy_plant_t = np.mean(ten_accuracy_plant_t)
        mean_weighted_plant_t = np.mean(ten_weighted_plant_t)
        mean_accuracy_disease_t = np.mean(ten_accuracy_disease_t)
        mean_weighted_disease_t = np.mean(ten_weighted_disease_t)
        
        # import scipy.stats as st
        std_acc_t = np.std(ten_accuracy_t, axis=0)
        std_f1_t = np.std(ten_weighted_t, axis=0)
        
        std_acc_plant_t = np.std(ten_accuracy_plant_t, axis=0)
        std_f1_plant_t = np.std(ten_weighted_plant_t, axis=0)
        std_acc_disease_t = np.std(ten_accuracy_disease_t, axis=0)
        std_f1_disease_t = np.std(ten_weighted_disease_t, axis=0)
        
        print('--------------------------------------------------')    
        print( 'Mean_Accuracy_plant_t' ,mean_accuracy_plant_t, ' std_acc_plant_t: ', str(std_acc_plant_t))
        print( 'Mean_F1_score_plant_t' ,mean_weighted_plant_t, ' std_f1_plant_t: ', str(std_f1_plant_t))
        print( 'Mean_Accuracy_disease_t' ,mean_accuracy_disease_t, ' std_acc_dis_t: ', str(std_acc_disease_t))
        print( 'Mean_F1_score_disease_t' ,mean_weighted_disease_t, ' std_f1_dis_t: ', str(std_f1_disease_t))
        print( 'Mean_Accuracy_total_t' ,mean_acc_t, ' std_acc_t: ', str(std_acc_t))
        print( 'Mean_F1_score_total_t' ,mean_weighted_t, ' std_f1_t: ', str(std_f1_t))          
        
        
        
    if saveornot == 'save':
        if obj == "new_model":
            with open(dir_save+obj+'_'+plus +'_hist_'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
                h.write(item+'_'+obj+'_'+plus+' 10 times | ' + str(model_name) +'  Mean_Accuracy: '+str(mean_acc)+' std_acc_plant: '+ str(std_acc)+ '  Mean_F1_score: '+str(mean_weighted)+' std_f1: '+str(std_f1)+'\n')    
                h.write('Mean_Accuracy_t: '+str(mean_acc_t)+' std_acc_plant_t: '+ str(std_acc_t)+ '  Mean_F1_score_t: '+str(mean_weighted_t)+' std_f1_t: '+str(std_f1_t)+'\n'+'\n')    
        
            with open(dir_save+obj+'_'+'_'+plus +'_hist_plant&dis'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
                h.write(item+'_'+obj+'_'+'_'+plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_plant: '+str(mean_accuracy_plant)+' std_acc_plant: '+ str(std_acc_plant)+ '  Mean_F1_score_plant: '+str(mean_weighted_plant)+' std_f1_plant: '+str(std_f1_plant)+'\n')    
                h.write(' 10 times_t | ' + str(model_name) +'  Mean_Accuracy_plant_t: '+str(mean_accuracy_plant_t)+' std_acc_plant_t: '+ str(std_acc_plant_t)+ '  Mean_F1_score_plant_t: '+str(mean_weighted_plant_t)+' std_f1_plant_t: '+str(std_f1_plant_t)+'\n'+'\n')    
        
                h.write(item+'_'+obj+'_'+'_'+plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_disease: '+str(mean_accuracy_disease)+' std_acc_disease: '+ str(std_acc_disease)+ '  Mean_F1_score_disease: '+str(mean_weighted_disease)+' std_f1_disease: '+str(std_f1_disease)+'\n')    
                h.write(' 10 times_t | ' + str(model_name) +'  Mean_Accuracy_disease_t: '+str(mean_accuracy_disease_t)+' std_acc_disease_t: '+ str(std_acc_disease_t)+ '  Mean_F1_score_disease_t: '+str(mean_weighted_disease_t)+' std_f1_disease_t: '+str(std_f1_disease_t)+'\n'+'\n')    
                
            np.savetxt(dir_save +'/'+item+'_'+obj+'_'+plus +'_' + str(model_name)+'_Accuracy_10_times.txt', ten_accuracy,fmt='%s',delimiter=',')
            np.savetxt(dir_save +'/'+item+'_'+obj+'_'+plus +'_' + str(model_name)+'_Weighted_10_times.txt', ten_weighted,fmt='%s',delimiter=',')
        
            np.savetxt(dir_save +'/'+item+'_'+obj+'_'+plus +'_' + str(model_name)+'_Accuracy_t_10_times.txt', ten_accuracy_t,fmt='%s',delimiter=',')
            np.savetxt(dir_save +'/'+item+'_'+obj+'_'+plus +'_' + str(model_name)+'_Weighted_t_10_times.txt', ten_weighted_t,fmt='%s',delimiter=',')
        else:        
            with open(dir_save+obj+'_' +'_hist_'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
                h.write(item+'_'+obj+'_' +' 10 times | ' + str(model_name) +'  Mean_Accuracy: '+str(mean_acc)+' std_acc_plant: '+ str(std_acc)+ '  Mean_F1_score: '+str(mean_weighted)+' std_f1: '+str(std_f1)+'\n'+'\n')    
        
            with open(dir_save+obj+'_' +'_hist_plant&dis'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
                h.write(item+'_'+obj+'_' +' 10 times | ' + str(model_name) +'  Mean_Accuracy_plant: '+str(mean_accuracy_plant)+' std_acc_plant: '+ str(std_acc_plant)+ '  Mean_F1_score_plant: '+str(mean_weighted_plant)+' std_f1_plant: '+str(std_f1_plant)+'\n'+'\n')    
                h.write(item+'_'+obj+'_' +' 10 times | ' + str(model_name) +'  Mean_Accuracy_disease: '+str(mean_accuracy_disease)+' std_acc_disease: '+ str(std_acc_disease)+ '  Mean_F1_score_disease: '+str(mean_weighted_disease)+' std_f1_disease: '+str(std_f1_disease)+'\n'+'\n')    
                
            np.savetxt(dir_save +'/'+item+'_'+obj+'_' + str(model_name)+'_Accuracy_10_times.txt', ten_accuracy,fmt='%s',delimiter=',')
            np.savetxt(dir_save +'/'+item+'_'+obj+'_' + str(model_name)+'_Weighted_10_times.txt', ten_weighted,fmt='%s',delimiter=',')
     
    print('All Done!')    

