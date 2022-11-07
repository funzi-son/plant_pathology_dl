# Plant Pathology Deep Learning

This is the code for paper "Deep Learning for Plant Identification and Disease  Classification from Leaf Images: Multi-prediction Approaches".

**Abstract**: Deep learning has been playing an important role in modern agriculture, especially in plant pathology using leaf images where convolutional neural networks (CNN) are attracting a lot of attention. In this paper, we start our study by surveying current deep learning approaches for plant identification and disease classification. We categorise the approaches into multi-model, multi-label, multi-output, and multi-task, in which different backbone CNNs can be employed. Furthermore, based on the survey of existing approaches in plant pathology and the study of available approaches in machine learning, we propose a new model named Generalised Stacking Multi-output CNN (GSMo-CNN). To investigate the effectiveness of different backbone CNNs and learning approaches, we conduct an intensive experiment on three benchmark datasets Plant Village, Plant Leaves, and PlantDoc. The experiment results demonstrate that InceptionV3 can be a good choice for a backbone CNN as its performance is better than AlexNet, VGG16, ResNet101, EfficientNet, MobileNet, and a custom CNN developed by us. Interestingly, there is empirical evidence to support the hypothesis that using a single model for both tasks can be comparable or better than using two models, one for each task. Finally, we show that the proposed GSMo-CNN achieves state-of-the-art performance on three benchmark datasets.

## Requirements
Python >= 3.7




## Usage



## Parameters in main.py
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| ‘item’                     |  Dataset name, i.e., ‘plant_village’, ‘plant_leaves’, ‘PlantDoc’ & ‘PlantDoc_original’. |
| ‘obj’                     |  .|
| `model_name`                     |  .|
| `saveornot`                     |  .|
| `fig_size`                     |  .|
| `INIT_LR`                     |  .|
| `op_z`                     |  .|
| `TF_weights`                     |  .|
| `bat_si`                     |  .|
| `epo`                     |  .|
| `times `                     |  .|
| `p_t_w `                     |  .|
| `d_t_w `                     |  .|
| `p_w`                     |  .|
| `d_w `                     |  .|
| `balance_weight`                     |  .|








