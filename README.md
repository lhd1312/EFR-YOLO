# EFR-YOLO
Real-time edge computing detection of rice diseases

<img width="1193" height="444" alt="image" src="https://github.com/user-attachments/assets/ed081872-d4c3-4e06-afe7-be6b297d28ec" />



## Required environment
torch==2.2.2 cuda==12.1

## File download 
weight
webgage：https://pan.baidu.com/s/1VNGmzciW8eOKSWq_Umfmog?pwd=rzh2  
password：rzh2

datasets
Data will be made available on request. 

## Training
1. Preparation of datasets  
dataset
 -images
   --test
   --train
   --val
 -labels
   --test
   --train
   --val
**You need to download the dataset before training, unzip it and put it in the dataset directory.**  

2. Processing of datasets   
python split_data.py
  
**You can use xml2txt.py and yolo2coco.py to format your dataset.**  

4. Training 
python train.py 

## Evaluation 
python get_FPS.py  
**get the evaluation results**    

## Predict   
python detect.py  

## Reference
github:https://github.com/z1069614715/objectdetection_script
Jocher, G., Qiu, J., & Chaurasia, A. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
