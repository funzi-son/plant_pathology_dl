# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:32:03 2023

@author: Jianping Yao

jianping.yao@utas.edu.au
"""

import tensorflow as tf
print("tensorflow version: ", tf.__version__)
from tensorflow.keras import layers,models,utils,optimizers,callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Dropout,Lambda,Dense,Flatten,Input
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse,random,pickle,cv2,os
import pandas as pd
import tensorflow_addons as tfa

# multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
DEVICES = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tip = "fixed_test"
# tip = "no_fixed"
plus = "_ViT_v1"

obj = "powerset" # Traditional Multi-label

item =  "plant_village"
# item = "plant_leaves"
# item =  "PlantDoc"
# item = "PlantDoc_original"

saveornot = 'save'
# saveornot = 'not'

bat_si = 8 #16   # batch_size
epo =  10000  #20  # epochs
times = 10
fig_size = 256
# 
server = "PC"
server = "jyao1"

if server == "PC":
    HDD = 'D' 
    load_data_dir = HDD +':/plant_pathology_dl/'
else:
    print('error')
import sys
sys.path.append(load_data_dir)
from .. import load_data
    
model_list = [
    'ViT' , 
    ]

########################
INIT_LR = 0.001 #0.0009 #  0.0009 is the best with Adamax so far (72% acc)
weight_decay = 0.0001
# op_z = "Adamax" # Adamax  AdamW

patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (fig_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [ projection_dim * 2, projection_dim]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
# mlp_head_units = [256, 128]  # Size of the dense layers of the final classifier


#################################################

## This code of the ViT was modified from:
## https://github.com/keras-team/keras-io/blob/54392950c4142c96ccc0f8dfd4a9a586edbe5cf2/examples/vision/image_classification_with_vision_transformer.py#L280
## https://keras.io/examples/vision/image_classification_with_vision_transformer/

"""
## Implement multilayer perceptron (MLP)
"""


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        ##############################################
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size
        })
        return config
    ############################################
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""
## Implement the patch encoding layer
The `PatchEncoder` layer will linearly transform a patch by projecting it into a
vector of size `projection_dim`. In addition, it adds a learnable position
embedding to the projected vector.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.prj_dim = projection_dim
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
            'projection_dim': self.prj_dim   
            })
        return config
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier(categ,fig_size):
    input_shape = (fig_size, fig_size, 3)
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    # patches = Patches(patch_size)(augmented)
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    
    # # Classify outputs.
    # logits = layers.Dense(num_classes)(features)
    # # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=logits)
    
    output_c = Dense(categ, "softmax")(features) #, name="total_output"
    model = Model(inputs=inputs, outputs=output_c)
    return model, inputs

for model_name in model_list:
    x = []
    y = []
    PlantLabels = []
    diseaseLabels = []
    if item == "PlantDoc_original":
        dataset, fig_size, model_dir, categ, dir_save,categories, p_type, d_type, test_set = load_data.load_data(model_name, server,item,obj,tip,fig_size)

        x_train = []
        y_train = []
        for features, label in dataset:
            x_train.append(features)
            y_train.append(label)
        x_train=np.array(x_train).reshape(-1,fig_size,fig_size,3)
        
        x_test = []
        y_test = []
        for features, label in test_set:
            x_test.append(features)
            y_test.append(label)
        x_test=np.array(x_test).reshape(-1,fig_size,fig_size,3)
        # y_test=np.array(y_test).reshape(-1,1)
        
        
    else:
        dataset, fig_size, model_dir, categ, dir_save,categories, p_type, d_type = load_data.load_data(model_name, server,item,obj,tip,fig_size)

        for features, label in dataset:
            x.append(features)
            y.append(label)
            # PlantLabels.append(Plant)
            # diseaseLabels.append(dis)
        x=np.array(x).reshape(-1,fig_size,fig_size,3)
 

        if tip == "fixed_test":

            split = train_test_split(x, y, test_size=0.2, random_state = 50)
            (x_train, x_test, y_train, y_test) = split
        else:
            pass
        # y_disease_test = np.argmax(testDiseaseY,axis=1)
        # y_plant_test = np.argmax(testPlantY,axis=1)
    # y_disease_test=[]
    # y_plant_test=[]
    # es = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=1, mode='max',restore_best_weights=True)
    # mc = callbacks.ModelCheckpoint(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
    
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min',restore_best_weights=True) #
    mc = callbacks.ModelCheckpoint(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)

    ten_accuracy = []
    ten_weighted = []
    ten_accuracy_plant = []
    ten_weighted_plant = []
    ten_accuracy_disease = []
    ten_weighted_disease = []
    
    # inputShape = (fig_size, fig_size, 3)
    # inputs = Input(shape=inputShape) #, name='main_input'

    # chanDim = -1
    
    for g in range(0,times):
        print(item,' ', obj, ' ', model_name )
        print('Runing ', g, ' time..............')
        
        if tip == "no_fixed":            
            split = train_test_split(x, y, test_size=0.2)
            (trainX, testX, y_train, y_test) = split
        elif tip == "fixed_test" or item == "PlantDoc_original":
            pass
        else:
            print("tip error!!!")

#################################################################################
        # split the data    save to pickle files
        
        K.clear_session()      
        
        if model_name == 'ViT':   

            # with strategy.scope():
            model, inputs = create_vit_classifier(categ, fig_size)
            # opt = eval(op_z)(learning_rate=INIT_LR) #, weight_decay=weight_decay)
            opt = tfa.optimizers.AdamW(learning_rate=INIT_LR, weight_decay=weight_decay)
            model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=["accuracy"])
            
        else:
            print("Model name error!")



        y_train_cat=tf.one_hot(y_train,categ)  #Tensorflow 2
        y_test_cat=tf.one_hot(y_test,categ)
        
        #fit the model i.e. training the model and batch size can be varies
        H=model.fit(x_train,y_train_cat,batch_size=bat_si, #16,
                  epochs=epo,verbose=1,
                  validation_split=0.1,
                   shuffle=True,
                   callbacks=[es,mc]
                  )

        # ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'.h5')
        # ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5')
        
        # loss, acc = ModelCheck_model.evaluate(x_test,y_test_cat, verbose=1)
        # predict_x=ModelCheck_model.predict(x_test)            # tf >= 2.6
        predict_x=model.predict(x_test)   
        y_pred=np.argmax(predict_x,axis=1)


        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score,f1_score
        accuracy_total = accuracy_score(y_test, y_pred)
        weighted_total = f1_score(y_test, y_pred, average='weighted')
        
        cm_total = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))

        

        y_disease_test=[]
        y_plant_test=[]
        z = 0
        for n in y_test:
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
        for l in y_pred:
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
 

        accuracy_plant = accuracy_score(y_plant_test, y_plant_pred)
        weighted_plant = f1_score(y_plant_test, y_plant_pred, average='weighted')
        cm_plant = confusion_matrix(y_plant_test, y_plant_pred)#, labels=categories)
        
        accuracy_disease = accuracy_score(y_disease_test, y_disease_pred)
        weighted_disease = f1_score(y_disease_test, y_disease_pred, average='weighted')
        cm_disease = confusion_matrix(y_disease_test, y_disease_pred)#, labels=categories)
            
            

        
        # cm_plant = confusion_matrix(y_plant_test, y_plant_pred)#, labels=categories)
        print( model_name+' | '+ item + '_'+ obj +' SKlearn Accuracy_total' ,accuracy_total,' SKlearn F1_score_total' ,weighted_total)#'\n',
        
   


        # model.save(args["model"], save_format="h5")
        model_history = H.history
        if saveornot == 'save':
            path_save = dir_save +'model_save/'+ model_name+'/'+obj+'_'+ tip +'_model_'+ str(accuracy_total)+'fig_size_' + str(fig_size)
            if not os.path.exists(path_save):
                print('Model & Result do not exist, make dir.')
                os.makedirs(path_save)
        
        plt.figure(1)
        plt.plot(model_history['accuracy'])
        plt.plot(model_history['val_accuracy'])
        plt.title(str(model_name)+"'s Accuracy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'])
        if saveornot == 'save':
            plt.savefig(path_save+'/Acc&Val_acc.png',dpi=1000)
        plt.show()
        
        plt.figure(2)
        plt.plot(model_history['loss'])
        plt.plot(model_history['val_loss'])
        plt.title(str(model_name)+"'s loss")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'])
        if saveornot == 'save':
            plt.savefig(path_save+'/Loss&Val_loss.png',dpi=1000)
        plt.show()

        import itertools 
        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,fonts = 5, rot = 45):
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')
        #    print(cm)
            plt.figure(  dpi=1000)# figsize=(12, 8),
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title,fontsize=(fonts*1.3))
            
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=rot,fontsize=fonts*1.3)
            plt.yticks(tick_marks, classes,fontsize=fonts)
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",fontsize=fonts)
            # plt.tight_layout()
            plt.ylabel('True label', fontsize=(fonts*1.3))
            plt.xlabel('Predicted label', fontsize=(fonts*1.3))  
            
        plt.figure(3)#,tight_layout=True) #, figsize=(200,200), dpi=1000
        
        labels_p=list(p_type)
        plot_confusion_matrix(cm_plant, classes=labels_p, normalize=False, title=item + '_'+obj+' | '+model_name+' Confusion Matrix',fonts = 9,rot =90) #+' Confusion Matrix'
        if saveornot == 'save':
            plt.savefig(path_save +'/'+item + '_'+obj+'_'+model_name+'_plant_confusion_matrix.jpg',dpi=1000, bbox_inches = 'tight')  #, bbox_inches = 'tight'
        plt.show()
        
        plt.figure(4)
        labels_d=list(range(len(d_type)))
        plot_confusion_matrix(cm_disease, classes=labels_d, normalize=False, title=item + '_'+obj+' | '+model_name+'_multi_tasks_disease ',fonts = 5)#+model_name+' Confusion Matrix'
        if saveornot == 'save':
            plt.savefig(path_save +'/'+item + '_'+obj+'_'+model_name+'_disease_confusion_matrix.jpg',dpi=1000, bbox_inches = 'tight')  #, bbox_inches = 'tight'
        plt.show()
        
        plt.figure(5)
        labels_t=list(range(len(set(categories))))
        # labels_t=sorted(set(categories))
        plot_confusion_matrix(cm_total, classes=labels_t, normalize=False, title=item + '_'+obj+' | '+model_name+'_multi_tasks_Total ',fonts = 5,rot =0)#+model_name+' Confusion Matrix'
        if saveornot == 'save':
            plt.savefig(path_save +'/'+item + '_'+obj+'_'+model_name+'_total_confusion_matrix.jpg',dpi=1000, bbox_inches = 'tight')  #, bbox_inches = 'tight'
        plt.show()
        
        pd.DataFrame(y_plant_test).to_csv(path_save +'/y_plant_test_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_plant_pred).to_csv(path_save +'/y_plant_pred_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_disease_test).to_csv(path_save +'/y_disease_test_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_disease_pred).to_csv(path_save +'/y_disease_pred_'+item + '_'+obj+'_'+model_name+'_.csv')
        
        pd.DataFrame(y_test).to_csv(path_save +'/y_total_test_t_'+item + '_'+obj+'_'+model_name+'_.csv')
        pd.DataFrame(y_pred).to_csv(path_save +'/y_total_pred_t_'+item + '_'+obj+'_'+model_name+'_.csv')   
        

        if saveornot == 'save':
            from tensorflow.keras.utils import plot_model
            plot_model(model, to_file = path_save+'/'+item + '_'+obj+'_'+model_name+'.png')
            

            np.savetxt(path_save +'/training_acc_hist.txt', model_history['accuracy'],fmt='%s',delimiter=',')
            np.savetxt(path_save +'/training_loss_hist.txt', model_history['loss'],fmt='%s',delimiter=',')
            
            np.savetxt(path_save +'/training_val_acc_hist.txt', model_history['val_accuracy'],fmt='%s',delimiter=',')
            np.savetxt(path_save +'/training_val_loss_hist.txt', model_history['val_loss'],fmt='%s',delimiter=',')    

            # np.savetxt(path_save +'/model_summary.txt', 'model.summary')
            with open(path_save + '/model_summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            with open(path_save +'/callback_params.txt',"w") as k:
                k.write( str(es.params) )
                # k.close()        
            # #saving the trained model so that no need to fit again for next time
            model.save(path_save+'/'+item + '_'+obj+'_'+model_name+'_multi_tasks.h5')
        
            print('Saved to ', path_save)
        # with open('/home/sntran/Projects/plant_disease_detection/Tuning_history/tuning_hist_plant_village_disease.txt', 'a') as h:
        with open(dir_save+item +'_'+ obj + '_'+model_name+'_hist.txt', 'a') as h:
            h.write(item +'_'+ obj + '_' + str(model_name) + '  fig_size: '+ str(fig_size)+ '  batch_size: '+str(bat_si)+
                    '  epochs: '+str(epo)+'  Accuracy_plant: '+str(accuracy_plant)+ '  F1_score_plant: '+str(weighted_plant)+
                    '  Accuracy_disease: '+str(accuracy_disease)+ '  F1_score_disease: '+str(weighted_disease)+
                    '  Accuracy_total: '+str(accuracy_total)+ '  F1_score_total: '+str(weighted_total)+'\n')    

        print( 'No.'+ str(g)+' total_Accuracy' ,accuracy_total)
        print( 'No.'+ str(g)+' total_F1_score' ,weighted_total)
        
        print( 'Accuracy_plant' ,accuracy_plant)#'\n',
        print( 'F1_score_plant' ,weighted_plant)#'\n',
        print( 'Accuracy_disease' ,accuracy_disease)#'\n',
        print( 'F1_score_disease' ,weighted_disease)#'\n',            
        
        
        ten_accuracy.append(accuracy_total)
        ten_weighted.append(weighted_total)
        
        ten_accuracy_plant.append(accuracy_plant)
        ten_weighted_plant.append(weighted_plant)
        ten_accuracy_disease.append(accuracy_disease)
        ten_weighted_disease.append(weighted_disease)
        
  
    
    mean_acc = np.mean(ten_accuracy)
    mean_weighted = np.mean(ten_weighted)
    
    mean_accuracy_plant = np.mean(ten_accuracy_plant)
    mean_weighted_plant = np.mean(ten_weighted_plant)
    mean_accuracy_disease = np.mean(ten_accuracy_disease)
    mean_weighted_disease = np.mean(ten_weighted_disease)
    
    import scipy.stats as st
    std_acc = np.std(ten_accuracy, axis=0)
    std_f1 = np.std(ten_weighted, axis=0)
    
    std_acc_plant = np.std(ten_accuracy_plant, axis=0)
    std_f1_plant = np.std(ten_weighted_plant, axis=0)
    std_acc_disease = np.std(ten_accuracy_disease, axis=0)
    std_f1_disease = np.std(ten_weighted_disease, axis=0)
    
    
    
    print('\n',model_name + ' | '+item+'_'+ obj+'_'+ tip  + " | Fig_size: ", fig_size, '*', fig_size, " Done!")
    
    print( 'Mean_Accuracy' ,mean_acc, ' std_acc: ', str(std_acc))
    print( 'Mean_F1_score' ,mean_weighted, ' std_f1: ', str(std_f1))

    print( 'Mean_Accuracy_plant' ,mean_accuracy_plant, ' std_acc_plant: ', str(std_acc_plant))
    print( 'Mean_F1_score_plant' ,mean_weighted_plant, ' std_f1_plant: ', str(std_f1_plant))
    print( 'Mean_Accuracy_disease' ,mean_accuracy_disease, ' std_acc_dis: ', str(std_acc_disease))
    print( 'Mean_F1_score_disease' ,mean_weighted_disease, ' std_f1_dis: ', str(std_f1_disease))
    
    with open(dir_save+obj+'_'+ tip +'_hist_'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
        h.write(item+'_'+obj+'_'+ tip +' 10 times | ' + str(model_name) +'  Mean_Accuracy: '+str(mean_acc)+' std_acc_plant: '+ str(std_acc)+ '  Mean_F1_score: '+str(mean_weighted)+' std_f1: '+str(std_f1)+'\n'+'\n')    

    with open(dir_save+obj+'_'+ tip +'_hist_plant&dis'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
        h.write(item+'_'+obj+'_'+ tip +' 10 times | ' + str(model_name) +'  Mean_Accuracy_plant: '+str(mean_accuracy_plant)+' std_acc_plant: '+ str(std_acc_plant)+ '  Mean_F1_score_plant: '+str(mean_weighted_plant)+' std_f1_plant: '+str(std_f1_plant)+'\n'+'\n')    
        h.write(item+'_'+obj+'_'+ tip +' 10 times | ' + str(model_name) +'  Mean_Accuracy_disease: '+str(mean_accuracy_disease)+' std_acc_disease: '+ str(std_acc_disease)+ '  Mean_F1_score_disease: '+str(mean_weighted_disease)+' std_f1_disease: '+str(std_f1_disease)+'\n'+'\n')    
        
    np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ tip +'_' + str(model_name)+'_Accuracy_10_times.txt', ten_accuracy,fmt='%s',delimiter=',')
    np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ tip +'_' + str(model_name)+'_Weighted_10_times.txt', ten_weighted,fmt='%s',delimiter=',')
        
    
print('All Done!')    
    
    
