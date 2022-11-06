# -*- coding: utf-8 -*-

# dataset_dir = '/home/jyao1/Leaf_diseases/Datasets/'
# save_path = '/home/jyao1/Leaf_diseases/test_model/'
dataset_dir = 'D:/IT_learning/ResearchMaster/ComputerVision/Data/'
save_path = 'D:/IT_learning/test_model/'

# TF_weights='imagenet' #
TF_weights=None

fig_size = 32 #256
INIT_LR = 0.001    # INIT_LR = 0.004 
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
epo = 2 #10000 # epochs
times = 1#0

# model_name =  'CNN'
# model_name ='AlexNet'
#'VGG', 'ResNet'
# model_name =      'EfficientNet' 
# model_name =    'Inception'
model_name =   'MobileNet'

p_t_w = 0.1
d_t_w = 0.1
p_w = 0.4
d_w = 0.5   
balance_weight = [p_t_w,d_t_w,p_w,d_w]


if __name__ == '__main__':
    import Leaf_disease_main
    Leaf_disease_main.main(item, obj, model_name,dataset_dir,save_path,saveornot,fig_size, bat_si, INIT_LR, epo, times, op_z, TF_weights, balance_weight)


