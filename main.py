# -*- coding: utf-8 -*-
'''
Plant Pathology Deep Learning

This is the code for paper "Deep Learning for Plant Identification and Disease Classification
from Leaf Images: Multi-prediction Approaches".
Please modify the parameters at here, then run main.py.
'''

dataset_dir = '/home/user/Leaf_diseases/Datasets/'  # please change the path of the datasets. 
save_path = '/home/user/Leaf_diseases/test_model/'    # please change the path of results and model will be saved.

# TF_weights='imagenet'
TF_weights=None

fig_size = 256
INIT_LR = 0.001
op_z = "Adamax"

# obj = "multi_model"
obj = "multi_output"
# obj = "new_model"
# obj = "multi_label"
# obj = "cross_stitch"

item =  "plant_village"
# item = "plant_leaves"
# item =  "PlantDoc"
# item = "PlantDoc_original"

saveornot = 'save'
# saveornot = 'not'

bat_si = 16   # batch_size
epo = 10000 # epochs
times = 10

# model_name =  'CNN'
# model_name ='AlexNet'
# model_name = 'VGG'
# model_name = 'ResNet'
# model_name = 'EfficientNet' 
model_name = 'Inception'
# model_name =   'MobileNet'

p_t_w = 0.1
d_t_w = 0.1
p_w = 0.4
d_w = 0.5   

balance_weight = [p_t_w,d_t_w,p_w,d_w]


if __name__ == '__main__':
    import Leaf_disease_main
    Leaf_disease_main.main(item, obj, model_name,dataset_dir,save_path,saveornot,fig_size, bat_si, INIT_LR, epo, times, op_z, TF_weights, balance_weight)


