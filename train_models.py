# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:38:39 2022

@author: jyao1
"""
from tensorflow.keras import layers,optimizers,callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Dropout,Lambda,Dense,Flatten,Input,MaxPool2D
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import *


def new_model(input_layer):
    x = Conv2D(32,(3,3), padding='same',activation='relu')(input_layer)
    x= (BatchNormalization(axis=1))(x)    
    x = Conv2D(32,(3,3), padding="same",activation='relu')(x) 
    x = (BatchNormalization(axis=1))(x)
    x = (MaxPool2D(pool_size=(8,8)))(x)
    x = (BatchNormalization(axis=1))(x)
    x = Conv2D(32,(3,3), padding='same',activation='relu')(x) 
    x = (BatchNormalization(axis=1))(x)
    x = Conv2D(32,(3,3), padding='same',activation='relu')(x)
    x = (BatchNormalization(axis=1))(x)
    x = (MaxPool2D(pool_size=(8,8)))(x)
    x = (BatchNormalization(axis=1))(x)
    x = Flatten()(x)
    return x


def multi_output_model(model_name,input_layer,numPlants,numDis,categ, item, w_p, w_d, w_p_t,w_d_t, TF_weights,obj,op_z,INIT_LR):

    if model_name == 'CNN':
        x = new_model(input_layer)
    elif model_name == 'AlexNet':   
        x= (layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'))(input_layer)
        x= (layers.BatchNormalization())(x)
        x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
        x= (layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))(x)
        x= (layers.BatchNormalization())(x)
        x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
        x= (layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
        x= (layers.BatchNormalization())(x)
        x= (layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
        x= (layers.BatchNormalization())(x)
        x= (layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
        x= (layers.BatchNormalization())(x)
        x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
        x= (layers.Flatten())(x)
        x= (layers.Dense(4096, activation='relu'))(x)
        x= (layers.Dropout(0.5))(x)
        x= (layers.Dense(4096, activation='relu'))(x)
        x= (layers.Dropout(0.5))(x)
        # x= (layers.Dense(numplants))(x)  #fig_size ,activation='relu'
        # x= Activation(finalAct, name="Plant_output")(x)
    
    else:
        if model_name == 'VGG':
            from tensorflow.keras.applications import VGG16
            base_model = VGG16(input_tensor=input_layer, weights=TF_weights, include_top=False)

        elif model_name == 'ResNet':
            # from tensorflow.keras.applications import ResNet50
            # base_model = ResNet50(input_tensor=inputShape, weights=TF_weights, include_top=False)
        
            from tensorflow.keras.applications import ResNet101
            base_model = ResNet101(input_tensor=input_layer, weights=TF_weights, include_top=False)
    
        elif model_name == 'EfficientNet':
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(input_tensor=input_layer, weights=TF_weights, include_top=False)
    
        elif model_name == 'Inception':
            from tensorflow.keras.applications import InceptionV3
            base_model = InceptionV3(input_tensor=input_layer, weights=TF_weights, include_top=False)  
    
        elif model_name == 'MobileNet':
            from tensorflow.keras.applications import MobileNetV2
            base_model = MobileNetV2(input_tensor=input_layer, weights=TF_weights, include_top=False)    
    
        else:
            print("Model name error!")
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        # x= (layers.Flatten())(x)
        # x= (layers.Dropout(Drop))(x)



    if obj == "multi_output":

        disease_branch = Dense(numDis, activation='softmax', name='disease_output')(x)
        plant_branch = Dense(numPlants, activation='softmax', name='plant_output')(x)

        model = Model(inputs = input_layer,
             outputs = [disease_branch, plant_branch])

        losses = {"disease_output": "categorical_crossentropy", "plant_output": "categorical_crossentropy",}
        lossWeights = {"disease_output": 1.0, "plant_output": 1.0}
        print("[INFO] compiling model...")
        # model.compile(optimizer='Adamax', loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
        opt = eval(op_z)(learning_rate=INIT_LR)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])


    elif obj == "new_model":
        
        output_plant_t = Dense(numPlants, "softmax", name="plant_output_t")(x)
        output_dis_t = Dense(numDis, "softmax", name="disease_output_t")(x)
        
        d_plant = layers.concatenate([x,output_plant_t])
        d_dis = layers.concatenate([x,output_dis_t])           
        
        output_plant = Dense(numPlants, "softmax", name="plant_output")(d_dis)
        output_dis = Dense(numDis, "softmax", name="disease_output")(d_plant)
        model = Model(inputs=input_layer, outputs=[output_plant, output_dis,output_plant_t, output_dis_t])
        
        losses = {"plant_output": "categorical_crossentropy",
        	"disease_output": "categorical_crossentropy",
            "plant_output_t": "categorical_crossentropy",
            	"disease_output_t": "categorical_crossentropy",}
        lossWeights = {"plant_output": w_p, "disease_output": w_d,"plant_output_t": w_p_t, "disease_output_t": w_d_t}
   
        opt = eval(op_z)(learning_rate=INIT_LR)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])

    elif obj == "multi_label":
        x = (layers.Dense(256,activation='relu'))(x)
        output_total = (layers.Dense(categ,activation='softmax'))(x)
        model = Model(inputs=input_layer, outputs=output_total)
        # model.compile(loss='categorical_crossentropy',optimizer='Adamax',metrics=['accuracy'])

        opt = eval(op_z)(learning_rate=INIT_LR)
        # model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model





def multi_model(model_name,input_layer,numPlants,numDis,item, TF_weights,op_z,INIT_LR):

#########################################################
    class PlantNet:
        def build_disease_branch(inputs, numDis,model_name,
                                 finalAct="softmax", chanDim=-1):
            if model_name == 'CNN':
                x= Conv2D(32,(3,3), padding='same')(inputs)
                x = Activation("relu")(x)
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.Conv2D(32,(3,3),activation='relu'))(x)
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.MaxPool2D(pool_size=(8,8)))(x)
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.Conv2D(32,(3,3),padding='same',activation='relu'))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.Conv2D(32,(3,3),activation='relu'))(x)    
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.MaxPool2D(pool_size=(8,8)))(x)    
                x= (layers.Activation('relu'))(x)    
                x= (layers.Flatten())(x)
                x= (layers.Dense(numDis))(x)  
                x= Activation(finalAct, name="disease_output")(x)
            elif model_name == 'AlexNet':
                x= (layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'))(inputs)
                x= (layers.BatchNormalization())(x)
                x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
                x= (layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
                x= (layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
                x= (layers.Flatten())(x)
                x= (layers.Dense(4096, activation='relu'))(x)
                x= (layers.Dropout(0.5))(x)
                x= (layers.Dense(4096, activation='relu'))(x)
                x= (layers.Dropout(0.5))(x)
                x= (layers.Dense(numDis))(x)  
                x= Activation(finalAct, name="disease_output")(x)
            else:
                print("Model name error!")
            return x
    
        def build_Plant_branch(inputs,
                               numplants,model_name, finalAct="softmax",
                               chanDim=-1):
            if model_name == 'CNN':
                x= Conv2D(32,(3,3), padding='same')(inputs)  
                x = Activation("relu")(x)
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.Conv2D(32,(3,3),activation='relu'))(x)
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.MaxPool2D(pool_size=(8,8)))(x)
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.Conv2D(32,(3,3),padding='same',activation='relu'))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.Conv2D(32,(3,3),activation='relu'))(x)    
                x= (layers.BatchNormalization(axis=chanDim))(x)
                x= (layers.MaxPool2D(pool_size=(8,8)))(x)    
                x= (layers.Activation('relu'))(x)    
                x= (layers.Flatten())(x)
                x= (layers.Dense(numplants))(x)  
                x= Activation(finalAct, name="plant_output")(x)
            elif model_name == 'AlexNet':
                x= (layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'))(inputs)
                x= (layers.BatchNormalization())(x)
                x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
                x= (layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
                x= (layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))(x)
                x= (layers.BatchNormalization())(x)
                x= (layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))(x)
                x= (layers.Flatten())(x)
                x= (layers.Dense(4096, activation='relu'))(x)
                x= (layers.Dropout(0.5))(x)
                x= (layers.Dense(4096, activation='relu'))(x)
                x= (layers.Dropout(0.5))(x)
                x= (layers.Dense(numplants))(x)  
                x= Activation(finalAct, name="plant_output")(x)
            else:
                print("Model name error!")
            return x
    
        def build(inputs, numDis, numPlants,model_name,
                  finalAct="softmax"):
            chanDim = -1
            diseaseBranch = PlantNet.build_disease_branch(inputs,
                                                            numDis,model_name, finalAct=finalAct, chanDim=chanDim)
            PlantBranch = PlantNet.build_Plant_branch(inputs,
                                                        numPlants,model_name, finalAct=finalAct, chanDim=chanDim)
            model = Model(
                inputs=inputs,
                outputs=[diseaseBranch, PlantBranch],
                name="PlantNet")
            return model
        
    #########################################################
    class PlantNet_2:
    
        def build_disease_branch(input_layer, numDis,base_model,
                                 finalAct="softmax", chanDim=-1 ):
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x= (layers.Dense(numDis))(x)  
            x_1= Activation(finalAct, name="disease_output")(x)
            return x_1
    
        def build_Plant_branch(input_layer, 
                               numplants, base_model, finalAct="softmax",
                               chanDim=-1):
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x= (layers.Dense(numplants))(x)  
            x_2= Activation(finalAct, name="plant_output")(x)
            return x_2
    
        def build(input_layer, numDis, numPlants, base_model, base_model_2,
                  finalAct="softmax"):
    
            chanDim = -1

            diseaseBranch = PlantNet_2.build_disease_branch(input_layer,
                                                            numDis, base_model, finalAct=finalAct, chanDim=chanDim)
            PlantBranch = PlantNet_2.build_Plant_branch(input_layer,
                                                        numPlants, base_model_2, finalAct=finalAct, chanDim=chanDim)
            model = Model(
                inputs=input_layer,
                outputs=[diseaseBranch, PlantBranch],
                name="PlantNet")
            return model

##########################################

    K.clear_session()      
    
    if model_name == 'CNN' or model_name == 'AlexNet':
        
    # initialize our network
        model = PlantNet.build(input_layer,
        	numDis,
        	numPlants,model_name=model_name,
        	finalAct="softmax")
    elif model_name == 'VGG' or model_name == 'ResNet' or model_name == 'EfficientNet' or model_name == 'Inception' or model_name == 'MobileNet':
        if model_name == 'VGG':
            from tensorflow.keras.applications import VGG16
            base_model = VGG16(input_tensor=input_layer, weights=TF_weights, include_top=False)
            base_model_2 = VGG16(input_tensor=input_layer, weights=TF_weights, include_top=False)
        
        elif model_name == 'ResNet':
            # from tensorflow.keras.applications import ResNet50
            # base_model = ResNet50(input_tensor=input_layer, weights=TF_weights, include_top=False)
            # base_model_2 = ResNet50(input_tensor=input_layer, weights=TF_weights, include_top=False)        
            from tensorflow.keras.applications import ResNet101
            base_model = ResNet101(input_tensor=input_layer, weights=TF_weights, include_top=False)
            base_model_2 = ResNet101(input_tensor=input_layer, weights=TF_weights, include_top=False)
            
        elif model_name == 'EfficientNet':
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(input_tensor=input_layer, weights=TF_weights, include_top=False)
            base_model_2 = EfficientNetB0(input_tensor=input_layer, weights=TF_weights, include_top=False)
        elif model_name == 'Inception':
            from tensorflow.keras.applications import InceptionV3
            base_model = InceptionV3(input_tensor=input_layer, weights=TF_weights, include_top=False)    
            base_model_2 = InceptionV3(input_tensor=input_layer, weights=TF_weights, include_top=False)   
        elif model_name == 'MobileNet':
            from tensorflow.keras.applications import MobileNetV2
            base_model = MobileNetV2(input_tensor=input_layer, weights=TF_weights, include_top=False)    
            base_model_2 = MobileNetV2(input_tensor=input_layer, weights=TF_weights, include_top=False)    
        else:
            print("Model name error!")
        
        for layer in base_model.layers:
            layer._name = layer._name + str('_C')
    
        for layer_2 in base_model_2.layers:
            layer_2._name = layer_2._name + str('_D')
        
    
        model = PlantNet_2.build(input_layer,
         	numDis,
         	numPlants, base_model=base_model,base_model_2=base_model_2,
         	finalAct="softmax")
    else:
        print("Wrong model name!")
    
    
    losses = {
    	"disease_output": "categorical_crossentropy",
    	"plant_output": "categorical_crossentropy",
    }
    lossWeights = {"disease_output": 1.0, "plant_output": 1.0}
    # opt = Adamax(learning_rate=INIT_LR, decay=INIT_LR / epo)
    # model.compile(optimizer='Adamax', loss=losses, loss_weights=lossWeights,
    # 	metrics=["accuracy"])
    opt = eval(op_z)(learning_rate=INIT_LR)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
  
    
    for i, w in enumerate(model.weights):
        split_name = w.name.split('/')
        new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i)
        model.weights[i]._handle_name = new_name


    return model




#####################################################
'''
 This code of the Cross-stitch was modified from:
# https://github.com/AmazaspShumik/mtlearn/blob/8dc623e354df604c062288d8306768f7465fda97/mtlearn/layers/cross_stitch_block.py
Their paper:
@InProceedings{Misra_2016_CVPR,
	author = {Misra, Ishan and Shrivastava, Abhinav and Gupta, Abhinav and Hebert, Martial},
	title = {Cross-Stitch Networks for Multi-Task Learning},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2016}
}


'''

def cross_stitch(model_name,input_layer,numPlants,numDis,item,op_z,INIT_LR):


    import tensorflow as tf
    from typing import List
    from tensorflow.keras.initializers import RandomUniform
    from tensorflow.keras.layers import Layer
    
    
    class CrossStitchBlock(Layer):
        """
        Cross-stitch block
        References
        ----------
        [1] Cross-stitch Networks for Multi-task Learning, (2017)
        Ishan Misra et al
        """
    
        def build(self,
                  batch_input_shape: List[tf.TensorShape]
                  ) -> None:
            stitch_shape = len(batch_input_shape)
    
            # initialize using random uniform distribution as suggested in ection 5.1
            self.cross_stitch_kernel = self.add_weight(shape=(stitch_shape, stitch_shape),
                                                       initializer=RandomUniform(0., 1.),
                                                       trainable=True,
                                                       name="cross_stitch_kernel")
    
            # normalize, so that each row will be convex linear combination,
            # here we follow recommendation in paper ( see section 5.1 )
            normalizer = tf.reduce_sum(self.cross_stitch_kernel,
                                       keepdims=True,
                                       axis=0)
            self.cross_stitch_kernel.assign(self.cross_stitch_kernel / normalizer)
    
        def call(self, inputs):
            """
            Forward pass through cross-stitch block
            Parameters
            ----------
            inputs: np.array or tf.Tensor
                List of task specific tensors
            """
            # vectorized cross-stitch unit operation
            x = tf.concat([tf.expand_dims(e, axis=-1) for e in inputs], axis=-1)
            stitched_output = tf.matmul(x, self.cross_stitch_kernel)
    
            outputs = [tf.gather(e, 0, axis=-1) for e in tf.split(stitched_output, len(inputs), axis=-1)]
            return outputs

    x_1 = Conv2D(32,(3,3), padding='same')(input_layer)
    x_1 = Conv2D(16, (3, 3), padding="same")(x_1) 
    x_1 = Activation("relu")(x_1)
    x_1 = (BatchNormalization(axis=1))(x_1)
    x_1 = (MaxPool2D(pool_size=(8,8)))(x_1)
    x_1 = (BatchNormalization(axis=1))(x_1)
    
    x_2 = Conv2D(32,(3,3), padding='same')(input_layer) 
    x_2 = Conv2D(16, (3, 3), padding="same")(x_2) 
    x_2 = Activation("relu")(x_2)
    x_2 = BatchNormalization(axis=1)(x_2)
    x_2 = MaxPool2D(pool_size=(8,8))(x_2)
    x_2 = BatchNormalization(axis=1)(x_2)
    
    cs_block_1 = CrossStitchBlock()([x_1, x_2])
    
    x_c_1 = Conv2D(32,(3,3), padding='same',activation='relu')(cs_block_1[0]) 
    x_c_1 = (BatchNormalization(axis=1))(x_c_1)
    x_c_1 = Conv2D(32,(3,3), padding='same',activation='relu')(x_c_1)
    x_c_1 = (BatchNormalization(axis=1))(x_c_1)
    x_c_1 = (MaxPool2D(pool_size=(8,8)))(x_c_1)
    layer_2_1 = (BatchNormalization(axis=1))(x_c_1)
    
    x_c_2 = Conv2D(32,(3,3), padding='same',activation='relu')(cs_block_1[1]) 
    x_c_2 = (BatchNormalization(axis=1))(x_c_2)
    x_c_2 = Conv2D(32,(3,3), padding='same',activation='relu')(x_c_2)
    x_c_2 = (BatchNormalization(axis=1))(x_c_2)
    x_c_2 = (MaxPool2D(pool_size=(8,8)))(x_c_2)
    layer_2_2 = (BatchNormalization(axis=1))(x_c_2)
    
    # layer_2 = [x_c(stitch) for stitch in cs_block_1]
    cs_block_2 = CrossStitchBlock()([layer_2_1,layer_2_2] )

    stitch_1 = Flatten()(cs_block_2[0])
    # stitch_1 = Dense(12,"relu")(stitch_1)
    output_plant = Dense(numPlants, "softmax", name="plant_output")(stitch_1)
    
    stitch_2 = Flatten()(cs_block_2[1])
    # stitch_2 = Dense(12,"relu")(stitch_2)
    output_dis = Dense(numDis, "softmax", name="disease_output")(stitch_2)

    model = Model(inputs=input_layer, outputs=[output_plant, output_dis])
    losses = {
    	"disease_output": "categorical_crossentropy",
    	"plant_output": "categorical_crossentropy",
    }
    lossWeights = {"disease_output": 1.0, "plant_output": 1.0}
    
    opt = eval(op_z)(learning_rate=INIT_LR)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
    return model, CrossStitchBlock




#####################################################















    

