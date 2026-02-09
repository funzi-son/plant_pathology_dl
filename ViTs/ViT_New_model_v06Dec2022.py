# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:02:14 2022

@author: Jianping Yao

jianping.yao@utas.edu.au
"""


import tensorflow as tf
print("tensorflow version: ", tf.__version__)
from tensorflow.keras import layers,models,utils,optimizers,callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Dropout,Lambda,Dense,Flatten,Input,MaxPool2D
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
DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

w_p = 0.4 
w_d = 0.5 
w_p_t = 0.1 
w_d_t = 0.1

# w_p = 0.1 
# w_d = 0.1 
# w_p_t = 0.1
# w_d_t = 0.1

# w_p = 0.1      # W3
# w_d = 0.4      # W4
# w_p_t = 0.4    # W2
# w_d_t = 0.5    # W1


plus = "_ViT_newmodel"

tip = "fixed_test"
# tip = "no_fixed"
fig_size = 256
obj = "multi_task"

item =  "plant_village"
# item = "plant_leaves"
# item =  "PlantDoc"
# item = "PlantDoc_original"

saveornot = 'save'
# saveornot = 'not'

bat_si = 8 #16   # batch_size
epo = 20000    # epochs
times = 20

drop = 0
INIT_LR = 0.0001 
# INIT_LR = 0.001 #0.0009 #  0.0009 is the best with Adamax so far (72% acc)
weight_decay = 0.0001
# op_z = "Adamax" # Adamax  AdamW

patch_size = 64  # Size of the patches to be extract from the input images
num_patches = (fig_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [ projection_dim * 2, projection_dim]  # Size of the transformer layers
transformer_layers = 8
# mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
mlp_head_units = [256, 128]  # Size of the dense layers of the final classifier

server = "PC"


if server == "PC" or server == "code_test":
    HDD = 'D' 
    load_data_dir = HDD +':/IT_Learning/ResearchMaster/git_lab/frank_leaf_test/Frank_projects/'
else:
    print('error')
import sys
sys.path.append(load_data_dir)
from .. import load_data

model_list = ['ViT']
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
    def __init__(self, patch_size, **kwargs):
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
    def __init__(self, num_patches, projection_dim, **kwargs):
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




def create_vit_classifier(p_type, d_type,fig_size):
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
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=drop)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=drop)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    # representation = layers.GlobalAveragePooling1D()(representation)

    # representation = layers.Dropout(0.1)(representation)  #0.5
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=drop) #0.5
    
    
    # # Classify outputs.
    # logits = layers.Dense(num_classes)(features)
    # # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=logits)
    

    output_plant_t = Dense(numPlants, "softmax", name="plant_output_t")(features)
    output_dis_t = Dense(numCategories, "softmax", name="disease_output_t")(features)

    d_plant = layers.concatenate([features,output_plant_t])
    d_dis = layers.concatenate([features,output_dis_t])           

    output_plant = Dense(numPlants, "softmax", name="plant_output")(d_dis)
    output_dis = Dense(numCategories, "softmax", name="disease_output")(d_plant)
    model = Model(inputs=inputs, outputs=[output_plant, output_dis,output_plant_t, output_dis_t])
    
    # model = Model(inputs=inputs, outputs=[output_p, output_d])

    return model

#################################################



for model_name in model_list:
    if item == "PlantDoc_original":
        trainX, trainDiseaseY, trainPlantY, fig_size, model_dir,  dir_save, p_type, d_type, testX, disease_y, plant_y = load_data.load_data(model_name, server,item,obj,tip,fig_size)
        trainX=np.array(trainX).reshape(-1,fig_size,fig_size,3)
        diseaseLB = LabelBinarizer()
        PlantLB = LabelBinarizer()
        trainDiseaseY = diseaseLB.fit_transform(trainDiseaseY)
        trainPlantY = PlantLB.fit_transform(trainPlantY)
        
        numCategories=len(diseaseLB.classes_)
        numPlants=len(PlantLB.classes_) 
        
        testX=np.array(testX).reshape(-1,fig_size,fig_size,3)
        disease_y = np.array(disease_y)
        plant_y = np.array(plant_y)
        

        diseaseLB_t = LabelBinarizer()
        PlantLB_t = LabelBinarizer()
        testDiseaseY = diseaseLB_t.fit_transform(disease_y)
        testPlantY = PlantLB_t.fit_transform(plant_y)
        

    else:
        data, diseaseLabels, PlantLabels, fig_size, model_dir, categ, dir_save, p_type, d_type  = load_data.load_data(model_name, server,item,obj,tip,fig_size)
        
        data=np.array(data).reshape(-1,fig_size,fig_size,3)
        
        diseaseLabels = np.array(diseaseLabels)
        PlantLabels = np.array(PlantLabels)

        diseaseLB = LabelBinarizer()
        PlantLB = LabelBinarizer()
        diseaseLabels = diseaseLB.fit_transform(diseaseLabels)
        PlantLabels = PlantLB.fit_transform(PlantLabels)
        numCategories=len(diseaseLB.classes_)
        numPlants=len(PlantLB.classes_)
    

        if tip == "fixed_test":

            split = train_test_split(data, diseaseLabels, PlantLabels, test_size=0.2, random_state = 50)
            (trainX, testX, trainDiseaseY, testDiseaseY, trainPlantY, testPlantY) = split
        else:
            pass

    
    # es = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=1, mode='max',restore_best_weights=True)
    # mc = callbacks.ModelCheckpoint(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
    
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto',restore_best_weights=True)
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
    # inputs = Input(shape=inputShape) #, name='main_input'


    for g in range(0,times):
        print(item,' ', obj, ' ', model_name )
        print('Runing ', g, ' time..............')
        
        if tip == "no_fixed":            
            split = train_test_split(data, diseaseLabels, PlantLabels, test_size=0.2)
            (trainX, testX, trainDiseaseY, testDiseaseY, trainPlantY, testPlantY) = split
        elif tip == "fixed_test" or item == "PlantDoc_original":
            pass
        else:
            print("tip error!!!")

#################################################################################

        K.clear_session()
        # K.clear_session()
        
        model = create_vit_classifier(p_type, d_type, fig_size)
            



        losses = {"plant_output": "categorical_crossentropy",
        	"disease_output": "categorical_crossentropy",
            "plant_output_t": "categorical_crossentropy",
            	"disease_output_t": "categorical_crossentropy",}
        lossWeights = {"plant_output": w_p, "disease_output": w_d,"plant_output_t": w_p_t, "disease_output_t": w_d_t}
        # initialize the optimizer and compile the model
        # print("[INFO] compiling model...")
        # opt = Adamax(learning_rate=INIT_LR, decay=INIT_LR / epo)
        # opt = Adamax(learning_rate=INIT_LR)
        opt = tfa.optimizers.AdamW(learning_rate=INIT_LR, weight_decay=weight_decay)
        # model.compile(optimizer='Adamax', loss=losses, loss_weights=lossWeights,
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
        	metrics=["accuracy"])



        H = model.fit(x=trainX,
                      y=[trainPlantY,trainDiseaseY,trainPlantY,trainDiseaseY],batch_size=bat_si,
                      # validation_data=(testX,{"disease_output": testDiseaseY, "plant_output": testPlantY}),
                      validation_split=0.1,
                      shuffle=True,
                      epochs=epo,
                      verbose=1,
                      callbacks=[es,mc
                                 ]
                      )

        # from tensorflow.keras.utils import plot_model
        # plot_model(model,show_shapes=True,show_dtype=True,show_layer_names=True)
        # from IPython.display import Image
        # Image(retina=True, filename='model.png')
        
        # ModelCheck_model=models.load_model(dir_save+'model_save/ModelCheckpoint/'+model_name+'_ModelCheck_'+str(fig_size)+'_d_'+DEVICES+'.h5', custom_objects={'Patches': Patches,'PatchEncoder':PatchEncoder})
        # (plant_y_pred,disease_y_pred,plant_y_pred_t,disease_y_pred_t)=ModelCheck_model.predict(testX)        
       

        (plant_y_pred,disease_y_pred,plant_y_pred_t,disease_y_pred_t)=model.predict(testX)

        
        y_disease_pred=np.argmax(disease_y_pred,axis=1)
        y_plant_pred=np.argmax(plant_y_pred,axis=1)
        
        
        y_disease_test = np.argmax(testDiseaseY,axis=1)
        y_plant_test = np.argmax(testPlantY,axis=1)
 
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score,f1_score
        
        accuracy_plant = accuracy_score(y_plant_test, y_plant_pred)
        weighted_plant = f1_score(y_plant_test, y_plant_pred, average='weighted')
        cm_plant = confusion_matrix(y_plant_test, y_plant_pred)#, labels=categories)
        
        accuracy_disease = accuracy_score(y_disease_test, y_disease_pred)
        weighted_disease = f1_score(y_disease_test, y_disease_pred, average='weighted')
        cm_disease = confusion_matrix(y_disease_test, y_disease_pred)#, labels=categories)

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

        cm_total = confusion_matrix(y_total_test, y_total_pred, labels=sorted(set(y_total_test)))
        
##########################################################################################################
        y_disease_pred_t=np.argmax(disease_y_pred_t,axis=1)
        y_plant_pred_t=np.argmax(plant_y_pred_t,axis=1)
        

        
        accuracy_plant_t = accuracy_score(y_plant_test, y_plant_pred_t)
        weighted_plant_t = f1_score(y_plant_test, y_plant_pred_t, average='weighted')
        cm_plant_t = confusion_matrix(y_plant_test, y_plant_pred_t)#, labels=categories)
        
        accuracy_disease_t = accuracy_score(y_disease_test, y_disease_pred_t)
        weighted_disease_t = f1_score(y_disease_test, y_disease_pred_t, average='weighted')
        cm_disease_t = confusion_matrix(y_disease_test, y_disease_pred_t)#, labels=categories)


        
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
            path_save = dir_save +'model_save/'+ model_name+'/'+obj+'_'+ tip+'_' + plus+'_No_'+str(g)+'/model_'+ str(round(weighted_total,5))+'fig_size_' + str(fig_size)
            if not os.path.exists(path_save):
                print('Model & Result do not exist, make dir.')
                os.makedirs(path_save)
        
        plt.figure(1,dpi=500)
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
            plt.savefig(path_save+'/Acc&Val_acc.png',dpi=1000)
        plt.show()
        
        
        plt.figure(2,dpi=500)
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
            plt.figure(dpi=200)# figsize=(12, 8),
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
            
        plt.figure(3,dpi=500)#,tight_layout=True) #, figsize=(200,200), dpi=1000
        labels_p=list(PlantLB.classes_)
        plot_confusion_matrix(cm_plant, classes=labels_p, normalize=False, title=(item + '_'+obj+' | '+model_name+' Confusion Matrix'+ ' | plant_F1_score: ' + str(round(weighted_plant,5))+'%'),fonts = 9,rot =90,) #+' Confusion Matrix'
        # plt.tight_layout()
        if saveornot == 'save':
            plt.savefig(path_save +'/'+item + '_'+obj+'_'+model_name+'_plant_confusion_matrix.jpg',dpi=200, bbox_inches = 'tight')  #, bbox_inches = 'tight'
        plt.show()
        

        plt.figure(4,dpi=500)
        labels_d=list(range(len(diseaseLB.classes_)))
        plot_confusion_matrix(cm_disease, classes=labels_d, normalize=False, title=item + '_'+obj+' | '+model_name+'_multi_tasks_disease'+ ' | disease_F1_score: ' + str(round(weighted_disease,5))+'%',fonts = 5)#+model_name+' Confusion Matrix'
        plt.tight_layout()
        if saveornot == 'save':
            plt.savefig(path_save +'/'+item + '_'+obj+'_'+model_name+'_disease_confusion_matrix.jpg',dpi=200, bbox_inches = 'tight')  #, bbox_inches = 'tight'
        plt.show()
        
        plt.figure(5,dpi=500)
        labels_t=list(range(len(set(y_total_test))))
        # labels_t=sorted(set(y_total_test_t))
        plot_confusion_matrix(cm_total, classes=labels_t, normalize=False, title=item + '_'+obj+' | '+model_name+'_multi_tasks_Total '+ '| total_F1_score: ' + str(round(weighted_total,5))+'%',fonts = 5,rot =90)#+model_name+' Confusion Matrix'
        plt.tight_layout()
        if saveornot == 'save':
            plt.savefig(path_save +'/'+item + '_'+obj+'_'+model_name+'_total_confusion_matrix.jpg',dpi=200, bbox_inches = 'tight')  #, bbox_inches = 'tight'
        plt.show()
        
       

        if saveornot == 'save':
            pd.DataFrame(y_plant_test).to_csv(path_save +'/y_plant_test_'+item + '_'+obj+'_'+model_name+'_.csv')
            pd.DataFrame(y_plant_pred).to_csv(path_save +'/y_plant_pred_'+item + '_'+obj+'_'+model_name+'_.csv')
            pd.DataFrame(y_disease_test).to_csv(path_save +'/y_disease_test_'+item + '_'+obj+'_'+model_name+'_.csv')
            pd.DataFrame(y_disease_pred).to_csv(path_save +'/y_disease_pred_'+item + '_'+obj+'_'+model_name+'_.csv')
            
            pd.DataFrame(y_total_test).to_csv(path_save +'/y_total_test_both_'+item + '_'+obj+'_'+model_name+'_.csv')
            pd.DataFrame(y_total_pred).to_csv(path_save +'/y_total_pred_both_'+item + '_'+obj+'_'+model_name+'_.csv')   
     
            pd.DataFrame(y_plant_pred_t).to_csv(path_save +'/y_plant_pred_t_'+item + '_'+obj+'_'+model_name+'_.csv')
            pd.DataFrame(y_disease_pred_t).to_csv(path_save +'/y_disease_pred_t_'+item + '_'+obj+'_'+model_name+'_.csv')
            pd.DataFrame(y_total_pred_t).to_csv(path_save +'/y_total_pred_t_'+item + '_'+obj+'_'+model_name+'_.csv')           
     
        
            from tensorflow.keras.utils import plot_model
            plot_model(model, to_file = path_save+'/'+item + '_'+obj+'_'+model_name+'.png',
                       show_shapes=True,show_dtype=True,show_layer_names=True)
            
            with open(path_save + '/model_summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            # with open(path_save +'/callback_params.txt',"w") as k:
            #     k.write( str(es.params) )
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


        print( 'No.'+ str(g))
        print( 'Accuracy_plant' ,accuracy_plant)#'\n',
        print( 'F1_score_plant' ,weighted_plant)#'\n',
        print( 'Accuracy_disease' ,accuracy_disease)#'\n',
        print( 'F1_score_disease' ,weighted_disease)#'\n',            
        print( 'total_Accuracy' ,accuracy_total)
        print( 'total_F1_score' ,weighted_total)        
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

###############################################################################################
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


    print('\n',model_name + ' | '+item+'_'+ obj+'_'+ tip+'_'+plus  + " | Fig_size: ", fig_size, '*', fig_size, " Done!")

    print( 'Mean_Accuracy_plant' ,mean_accuracy_plant, ' std_acc_plant: ', str(std_acc_plant))
    print( 'Mean_F1_score_plant' ,mean_weighted_plant, ' std_f1_plant: ', str(std_f1_plant))
    print( 'Mean_Accuracy_disease' ,mean_accuracy_disease, ' std_acc_dis: ', str(std_acc_disease))
    print( 'Mean_F1_score_disease' ,mean_weighted_disease, ' std_f1_dis: ', str(std_f1_disease))
    print( 'Mean_Accuracy_total' ,mean_acc, ' std_acc: ', str(std_acc))
    print( 'Mean_F1_score_total' ,mean_weighted, ' std_f1: ', str(std_f1))  
    print('--------------------------------------------------')    
    print( 'Mean_Accuracy_plant_t' ,mean_accuracy_plant_t, ' std_acc_plant_t: ', str(std_acc_plant_t))
    print( 'Mean_F1_score_plant_t' ,mean_weighted_plant_t, ' std_f1_plant_t: ', str(std_f1_plant_t))
    print( 'Mean_Accuracy_disease_t' ,mean_accuracy_disease_t, ' std_acc_dis_t: ', str(std_acc_disease_t))
    print( 'Mean_F1_score_disease_t' ,mean_weighted_disease_t, ' std_f1_dis_t: ', str(std_f1_disease_t))
    print( 'Mean_Accuracy_total_t' ,mean_acc_t, ' std_acc_t: ', str(std_acc_t))
    print( 'Mean_F1_score_total_t' ,mean_weighted_t, ' std_f1_t: ', str(std_f1_t))          
        
        
        
    if saveornot == 'save':
        with open(dir_save+obj+'_'+ tip+'_'+plus +'_hist_'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
            h.write(item+'_'+obj+'_'+ tip+'_'+plus+' 10 times | ' + str(model_name) +'  Mean_Accuracy: '+str(mean_acc)+' std_acc_plant: '+ str(std_acc)+ '  Mean_F1_score: '+str(mean_weighted)+' std_f1: '+str(std_f1)+'\n')    
            h.write('Mean_Accuracy_t: '+str(mean_acc_t)+' std_acc_plant_t: '+ str(std_acc_t)+ '  Mean_F1_score_t: '+str(mean_weighted_t)+' std_f1_t: '+str(std_f1_t)+'\n'+'\n')    
    
        with open(dir_save+obj+'_'+ tip+'_'+plus +'_hist_plant&dis'+item+'_DEVICES_'+DEVICES+'.txt', 'a') as h:
            h.write(item+'_'+obj+'_'+ tip+'_'+plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_plant: '+str(mean_accuracy_plant)+' std_acc_plant: '+ str(std_acc_plant)+ '  Mean_F1_score_plant: '+str(mean_weighted_plant)+' std_f1_plant: '+str(std_f1_plant)+'\n')    
            h.write(' 10 times_t | ' + str(model_name) +'  Mean_Accuracy_plant_t: '+str(mean_accuracy_plant_t)+' std_acc_plant_t: '+ str(std_acc_plant_t)+ '  Mean_F1_score_plant_t: '+str(mean_weighted_plant_t)+' std_f1_plant_t: '+str(std_f1_plant_t)+'\n'+'\n')    

            h.write(item+'_'+obj+'_'+ tip+'_'+plus +' 10 times | ' + str(model_name) +'  Mean_Accuracy_disease: '+str(mean_accuracy_disease)+' std_acc_disease: '+ str(std_acc_disease)+ '  Mean_F1_score_disease: '+str(mean_weighted_disease)+' std_f1_disease: '+str(std_f1_disease)+'\n')    
            h.write(' 10 times_t | ' + str(model_name) +'  Mean_Accuracy_disease_t: '+str(mean_accuracy_disease_t)+' std_acc_disease_t: '+ str(std_acc_disease_t)+ '  Mean_F1_score_disease_t: '+str(mean_weighted_disease_t)+' std_f1_disease_t: '+str(std_f1_disease_t)+'\n'+'\n')    
            
        np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ tip+'_'+plus +'_' + str(model_name)+'_Accuracy_10_times.txt', ten_accuracy,fmt='%s',delimiter=',')
        np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ tip+'_'+plus +'_' + str(model_name)+'_Weighted_10_times.txt', ten_weighted,fmt='%s',delimiter=',')

        np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ tip+'_'+plus +'_' + str(model_name)+'_Accuracy_t_10_times.txt', ten_accuracy_t,fmt='%s',delimiter=',')
        np.savetxt(dir_save +'/'+item+'_'+obj+'_'+ tip+'_'+plus +'_' + str(model_name)+'_Weighted_t_10_times.txt', ten_weighted_t,fmt='%s',delimiter=',')
            
    
print('All Done!')    
    
