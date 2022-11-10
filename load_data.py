# -*- coding: utf-8 -*-
import cv2,os,pickle

print('Loading data...')
def save_dataset(data,diseaseLabels,PlantLabels,dir_save,fig_size):
    pickle_out=open(dir_save+"original/data_"+str(fig_size)+".pickle","wb")
    pickle.dump(data,pickle_out,protocol = 4)
    pickle_out.close()

    pickle_out=open(dir_save+"original/diseaseLabels_"+str(fig_size)+".pickle","wb")
    pickle.dump(diseaseLabels,pickle_out,protocol = 4)
    pickle_out.close
    
    pickle_out=open(dir_save+"original/PlantLabels_"+str(fig_size)+".pickle","wb")
    pickle.dump(PlantLabels,pickle_out,protocol = 4)
    pickle_out.close
    
def load_data(dataset_dir,save_dir, model_name, item, obj, fig_size):

    print(model_name, 'Fig_size: ', fig_size)
    
    dir_save= str(save_dir) + str(item)+'_'+str(obj) +'/'

    global dirtrain
    global dirtest
    
    if item == "plant_village":    
        dirtrain = dataset_dir + 'plant_village/Plant_leave_diseases_dataset_without_augmentation/'      
    elif item == "plant_leaves":          
        dirtrain = dataset_dir + 'plant_leaves/'
    elif item == "PlantDoc":  
        dirtrain = dataset_dir + 'PlantDoc_Dataset/train'
    elif item == "PlantDoc_original":  
        dirtrain = dataset_dir + 'PlantDoc_original/train'
        dirtest  = dataset_dir + 'PlantDoc_original/test'

    
    if not os.path.exists(dir_save+'model_save/ModelCheckpoint/'):
        print('ModelCheckpoint folder does not exist, make dir.')
        os.makedirs(dir_save+'model_save/ModelCheckpoint/')
    if not os.path.exists(dir_save+'original/'):
        print('original folder does not exist, make dir.')
        os.makedirs(dir_save+'original/')    
    
    
    categories=sorted(os.listdir(dirtrain))
    print(categories)
    categ = len(categories)

    dataset = []
    diseaseLabels = []
    PlantLabels = []        
    p_type_t = []
    d_type_t = []    
    p_type = []
    d_type = []
    if obj == "multi_label":       
        if item == "plant_village":    
            for n in categories:
                n_p_type = n[:n.index("___")]
                # print(n_type)
                p_type_t.append(n_p_type)
                n_d_type = (n[n.index("___")+3:])
                d_type_t.append(n_d_type)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))
              
            print(p_type,'\n')  
            print(d_type,'\n')        
            try:

                pickle_in_data=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","rb")
                print('Loading dataset from pickle files...')
                dataset=pickle.load(pickle_in_data)
         
            except:
                for n in categories:
                    path=os.path.join(dirtrain,n)
                    class_num=categories.index(n)
                    print(n,'  ',class_num)
                    for i in os.listdir(path):
                
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path) #creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            dataset.append([img_array_fs,class_num])
                        except Exception as e:
                            print('Exception occurred',img_path)
                            pass
                print('Saving dataset as pickle files...')
                pickle_out=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","wb")
                pickle.dump(dataset,pickle_out,protocol = 4)
                pickle_out.close
    
    
    
        elif item == "plant_leaves": 
            for n in categories:
                n_p_type = n[:n.index("_")]
                # print(n_type)
                p_type_t.append(n_p_type)
                
                n_d_type = (n[n.index("_")+1:])
                print(n_d_type)
                if n_d_type == 'healthy':
                    d_type_t.append(n_d_type)
                else:
                    d_type_t.append(n)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))
            
            print(p_type,'\n')  
            print(d_type,'\n')  

            try:
                print('Loading dataset from pickle files...')
                pickle_in_data=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","rb")
                dataset=pickle.load(pickle_in_data)
            except:                
                for n in categories:
                    path=os.path.join(dirtrain,n)
                    class_num=categories.index(n)
                    print(n,'  ',class_num)
                    for i in os.listdir(path):
                        try:
                            img_array=cv2.imread(os.path.join(path,i))
                            img_array = cv2.resize(img_array, (fig_size, fig_size))
                            dataset.append([img_array,class_num])
                        except Exception as e:
                            print('Exception occurred',img_path)
                            pass
                print('Saving dataset as pickle files...')
                pickle_out=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","wb")
                pickle.dump(dataset,pickle_out,protocol = 4)
                pickle_out.close
                
        elif item == "PlantDoc":  
            for n in categories:
                n_p_type = n[:n.index(" ")]
                p_type_t.append(n_p_type)
                
                n_d_type = (n[n.index(" ")+1:])
            
                if n_d_type == 'leaf':
                    d_type_t.append('healthy')
                    print(n_p_type,'   ','healthy')
                elif n_d_type == 'Early blight leaf':
                    d_type_t.append('leaf early blight')
                    print(n_p_type,'   ','leaf early blight')
                else:
                    d_type_t.append(n_d_type)
                    print(n_p_type,'   ',n_d_type)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))
            print(p_type,'\n')  
            print(d_type,'\n')  
            try:
                print('Loading dataset from pickle files...')
                pickle_in_data=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","rb")
                dataset=pickle.load(pickle_in_data)
            except:                            
                for n in categories:
                    path=os.path.join(dirtrain,n)
                    class_num=categories.index(n)
                    print(n,'  ',class_num)
                    for i in os.listdir(path):
        
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path)#creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            dataset.append([img_array_fs,class_num])
                        except Exception as e:
                            print('Exception occurred',img_path)
                            pass
                print('Saving dataset as pickle files...')
                pickle_out=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","wb")
                pickle.dump(dataset,pickle_out,protocol = 4)
                pickle_out.close
            
        elif item == "PlantDoc_original":  
            for n in categories:
                n_p_type = n[:n.index(" ")]
                p_type_t.append(n_p_type)
                
                n_d_type = (n[n.index(" ")+1:])
            
                if n_d_type == 'leaf':
                    d_type_t.append('healthy')
                    print(n_p_type,'   ','healthy')
                elif n_d_type == 'Early blight leaf':
                    d_type_t.append('leaf early blight')
                    print(n_p_type,'   ','leaf early blight')
                else:
                    d_type_t.append(n_d_type)
                    print(n_p_type,'   ',n_d_type)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))
            print(p_type,'\n')  
            print(d_type,'\n')  
            try:
                print('Loading dataset from pickle files...')
                pickle_in_data=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","rb")
                dataset=pickle.load(pickle_in_data)
                print('Loading testaset from pickle files...')
                pickle_in_data=open(dir_save+"original/test_set_"+str(fig_size)+".pickle","rb")
                test_set=pickle.load(pickle_in_data)
            except:                            
                for n in categories:
                    path=os.path.join(dirtrain,n)
                    class_num=categories.index(n)
                    print(n,'  ',class_num)
                    for i in os.listdir(path):
        
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path)#creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            dataset.append([img_array_fs,class_num])
                        except Exception as e:
                            print('Exception occurred',img_path)#, value:',e.value)
                            pass
                import random
                random.Random(50).shuffle(dataset)
                print('Saving dataset as pickle files...')
                pickle_out=open(dir_save+"original/dataset_"+str(fig_size)+".pickle","wb")
                pickle.dump(dataset,pickle_out,protocol = 4)
                pickle_out.close
                       
                test_set = []
                
                for n in categories:
                    # print(n)
                    path=os.path.join(dirtest,n)
                    class_num=categories.index(n)
                    print(n,'  ',class_num)
                    for i in os.listdir(path):
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path)#creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            test_set.append([img_array_fs,class_num])
                        except Exception as e:
                            print('Exception occurred',img_path)#, value:',e.value)
                            pass
                

                print('Saving test_set as pickle files...')
                pickle_out=open(dir_save+"original/test_set_"+str(fig_size)+".pickle","wb")
                pickle.dump(test_set,pickle_out,protocol = 4)
                pickle_out.close
                

        else:
            print('Item error!')

    elif obj == "multi_output" or obj == "multi_model"or obj == "classifier_chains"or obj == "cross_stitch" or obj == "new_model"or obj == "MTAN"or obj == "TSNs"or obj == "MOON":     

        data = []
  
        if item == "plant_village": 
            for n in categories:
                n_p_type = n[:n.index("___")]
                # print(n_type)
                p_type_t.append(n_p_type)
                n_d_type = (n[n.index("___")+3:])
                # print(n_type)
                d_type_t.append(n_d_type)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))
            
            # categ = len(p_type)    
            print(p_type,'\n')  
            print(d_type,'\n')        
            try:
                print('Loading dataset from pickle files...')
                pickle_in_data=open(dir_save+"original/data_"+str(fig_size)+".pickle","rb")
                data=pickle.load(pickle_in_data)
                pickle_in_diseaseLabels=open(dir_save+"original/diseaseLabels_"+str(fig_size)+".pickle","rb")
                diseaseLabels=pickle.load(pickle_in_diseaseLabels)    
                pickle_in_PlantLabels=open(dir_save+"original/PlantLabels_"+str(fig_size)+".pickle","rb")
                PlantLabels=pickle.load(pickle_in_PlantLabels)               
            except:

                for n in categories:
                    # print(n)
                    path=os.path.join(dirtrain,n)
                    Plant=n[:n.index("___")]
                    # print(n)
                    dis = (n[n.index("___")+3:])
                    # print(d_type)
                    print(n,'  ',Plant,'  ',dis)
                    for i in os.listdir(path):
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path) #creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            # dataset.append([img_array_fs,Plant,dis])
                            diseaseLabels.append(dis)
                            PlantLabels.append(Plant)
                            data.append(img_array_fs)
                        except Exception as e:
                            print('Exception occurred',img_path)#, value:',e.value)
                            pass       
                print('Saving dataset as pickle files...')
                save_dataset(data,diseaseLabels,PlantLabels,dir_save,fig_size)
            
        elif item == "plant_leaves":   
            for n in categories:
                n_p_type = n[:n.index("_")]
                # print(n_type)
                p_type_t.append(n_p_type)
                
                n_d_type = (n[n.index("_")+1:])
                print(n_d_type)
                if n_d_type == 'healthy':
                    d_type_t.append(n_d_type)
                else:
                    d_type_t.append(n)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))

            # categ = len(p_type)    
            print(p_type,'\n')  
            print(d_type,'\n')  
            try:
                print('Loading dataset from pickle files...')
                pickle_in_data=open(dir_save+"original/data_"+str(fig_size)+".pickle","rb")
                data=pickle.load(pickle_in_data)
                pickle_in_diseaseLabels=open(dir_save+"original/diseaseLabels_"+str(fig_size)+".pickle","rb")
                diseaseLabels=pickle.load(pickle_in_diseaseLabels)    
                pickle_in_PlantLabels=open(dir_save+"original/PlantLabels_"+str(fig_size)+".pickle","rb")
                PlantLabels=pickle.load(pickle_in_PlantLabels)               
            except:

    
    
                for n in categories:
                    path=os.path.join(dirtrain,n)
                    Plant=n[:n.index("_")]
                    if n[n.index("_")+1:] == 'healthy':
                        dis=(n[n.index("_")+1:])
                    else:
                        dis=(n)
    
                    print(n,'  ',Plant,'  ',dis)
                    for i in os.listdir(path):
    
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path)#creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            # dataset.append([img_array_fs,Plant,dis])
                            diseaseLabels.append(dis)
                            PlantLabels.append(Plant)
                            data.append(img_array_fs)
                        except Exception as e:
                            print('Exception occurred',img_path)#, value:',e.value)
                            pass
                print('Saving dataset as pickle files...')
                save_dataset(data,diseaseLabels,PlantLabels,dir_save,fig_size)
        elif item == "PlantDoc":  
            for n in categories:
                n_p_type = n[:n.index(" ")]
                p_type_t.append(n_p_type)
                
                n_d_type = (n[n.index(" ")+1:])
            
                if n_d_type == 'leaf':
                    d_type_t.append('healthy')
                    print(n_p_type,'   ','healthy')
                elif n_d_type == 'Early blight leaf':
                    d_type_t.append('leaf early blight')
                    print(n_p_type,'   ','leaf early blight')
                else:
                    d_type_t.append(n_d_type)
                    print(n_p_type,'   ',n_d_type)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))
            print(p_type,'\n')  
            print(d_type,'\n')  
            try:
                print('Loading dataset from pickle files...')
                pickle_in_data=open(dir_save+"original/data_"+str(fig_size)+".pickle","rb")
                data=pickle.load(pickle_in_data)
                pickle_in_diseaseLabels=open(dir_save+"original/diseaseLabels_"+str(fig_size)+".pickle","rb")
                diseaseLabels=pickle.load(pickle_in_diseaseLabels)    
                pickle_in_PlantLabels=open(dir_save+"original/PlantLabels_"+str(fig_size)+".pickle","rb")
                PlantLabels=pickle.load(pickle_in_PlantLabels)               
            except:

                for n in categories:
                    path=os.path.join(dirtrain,n)
                
                    Plant = n[:n.index(" ")]
                    if n[n.index(" ")+1:] == 'leaf':
                       dis=('healthy')
                    elif n[n.index(" ")+1:] == 'Early blight leaf':
                        dis=('leaf early blight')
                    else:
                        dis=(n[n.index(" ")+1:])
                
                    print(n,' | ',Plant,' | ',dis)
                    
                    for i in os.listdir(path):
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path)#creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            # dataset.append([img_array_fs,Plant,dis])
                            diseaseLabels.append(dis)
                            PlantLabels.append(Plant)
                            data.append(img_array_fs)
                        except Exception as e:
                            print('Exception occurred',img_path)#, value:',e.value)
                            pass
                print('Saving dataset as pickle files...')
                save_dataset(data,diseaseLabels,PlantLabels,dir_save,fig_size)
                
        elif item == "PlantDoc_original":
            for n in categories:
                n_p_type = n[:n.index(" ")]
                p_type_t.append(n_p_type)
                
                n_d_type = (n[n.index(" ")+1:])
            
                if n_d_type == 'leaf':
                    d_type_t.append('healthy')
                    print(n_p_type,'   ','healthy')
                elif n_d_type == 'Early blight leaf':
                    d_type_t.append('leaf early blight')
                    print(n_p_type,'   ','leaf early blight')
                else:
                    d_type_t.append(n_d_type)
                    print(n_p_type,'   ',n_d_type)
            p_type = sorted(set(p_type_t))
            d_type = sorted(set(d_type_t))
            # print(p_type,'\n')  
            # print(d_type,'\n')  
            try:
                print('Loading dataset from pickle files...')
                pickle_in_data=open(dir_save+"original/data_"+str(fig_size)+".pickle","rb")
                trainX=pickle.load(pickle_in_data)
                pickle_in_diseaseLabels=open(dir_save+"original/diseaseLabels_"+str(fig_size)+".pickle","rb")
                trainDiseaseY=pickle.load(pickle_in_diseaseLabels)    
                pickle_in_PlantLabels=open(dir_save+"original/PlantLabels_"+str(fig_size)+".pickle","rb")
                trainPlantY=pickle.load(pickle_in_PlantLabels)   
        
                pickle_in_testX=open(dir_save+"original/testX_"+str(fig_size)+".pickle","rb")
                testX=pickle.load(pickle_in_testX)
                pickle_in_disease_y=open(dir_save+"original/disease_y_"+str(fig_size)+".pickle","rb")
                disease_y=pickle.load(pickle_in_disease_y)    
                pickle_in_plant_y=open(dir_save+"original/plant_y_"+str(fig_size)+".pickle","rb")
                plant_y=pickle.load(pickle_in_plant_y)                  


            except:

                dataset = []
    
                for n in categories:
                    path=os.path.join(dirtrain,n)
                
                    Plant = n[:n.index(" ")]
                    
                    if n[n.index(" ")+1:] == 'leaf':
                       dis=('healthy')
                    elif n[n.index(" ")+1:] == 'Early blight leaf':
                        dis=('leaf early blight')
                    else:
                        dis=(n[n.index(" ")+1:])
                
                    print(n,' | ',Plant,' | ',dis)
                    
                    for i in os.listdir(path):
                
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path)#creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            dataset.append([img_array_fs,Plant,dis])
                            # diseaseLabels.append(dis)
                            # PlantLabels.append(Plant)
                            # data.append(img_array_fs)
                        except Exception as e:
                            print('Exception occurred',img_path)#, value:',e.value)
                            pass
    
                trainX = []
                trainPlantY = [] 
                trainDiseaseY = []
                import random
                random.Random(50).shuffle(dataset)
                for features, label_1, label_2 in dataset:
                    trainX.append(features)
                    trainPlantY.append(label_1)
                    trainDiseaseY.append(label_2)   
    
    
                testset = []
                for n in categories:
                    path=os.path.join(dirtest,n)   
                    Plant = n[:n.index(" ")]
                    if n[n.index(" ")+1:] == 'leaf':
                       dis=('healthy')
                    elif n[n.index(" ")+1:] == 'Early blight leaf':
                        dis=('leaf early blight')
                    else:
                        dis=(n[n.index(" ")+1:])
                    # print(n,' | ',Plant,' | ',dis)
                    
                    for i in os.listdir(path):
                        try:
                            img_path = os.path.join(path,i)
                            img_array=cv2.imread(img_path)#creating the path of each image
                            img_array_fs = cv2.resize(img_array, (fig_size, fig_size))
                            testset.append([img_array_fs,Plant,dis])
                            # diseaseLabels.append(dis)
                            # PlantLabels.append(Plant)
                            # data.append(img_array_fs)
                        except Exception as e:
                            print('Exception occurred',img_path)#, value:',e.value)
                            pass
                
                testX = []
                disease_y = []
                plant_y = [] 
                # random.shuffle(dataset)
                for features, label_1, label_2 in testset:
                    testX.append(features)
                    plant_y.append(label_1)    
                    disease_y.append(label_2)
                
                print('Saving dataset as pickle files...')
                save_dataset(trainX,trainDiseaseY,trainPlantY,dir_save,fig_size)
                pickle_out=open(dir_save+"original/testX_"+str(fig_size)+".pickle","wb")
                pickle.dump(testX,pickle_out,protocol = 4)
                pickle_out.close()
            
                pickle_out=open(dir_save+"original/disease_y_"+str(fig_size)+".pickle","wb")
                pickle.dump(disease_y,pickle_out,protocol = 4)
                pickle_out.close
    
                pickle_out=open(dir_save+"original/plant_y_"+str(fig_size)+".pickle","wb")
                pickle.dump(plant_y,pickle_out,protocol = 4)
                pickle_out.close
                
        else:
            print('Item error!')


    if obj == "multi_label":   
        if item == "PlantDoc_original":        
            return dataset, categ, dir_save,categories, p_type, d_type, test_set
        else:
            return dataset, categ, dir_save, categories, p_type, d_type
    elif obj == "multi_output" or obj == "multi_model"or obj == "classifier_chains"or obj == "cross_stitch" or obj == "new_model" or obj == "MTAN"or obj == "TSNs"or obj == "MOON": 
        if item == "PlantDoc_original":
            return trainX, trainDiseaseY, trainPlantY,  categ, dir_save, p_type, d_type, testX, disease_y, plant_y 
        else:
            return data, diseaseLabels, PlantLabels,  categ, dir_save, p_type, d_type 
    else:
        print('Object error!')









