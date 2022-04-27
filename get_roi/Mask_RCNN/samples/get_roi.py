import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch
import csv
import ipdb
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Create model object in inference mode.
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


news_path = "/workspaceblobstore/data/v-chaozhang/mmrec/data/subnews.tsv"
roi_file = "/workspaceblobstore/data/v-chaozhang/mmrec/data/rois_np"

res50 = models.resnet50(pretrained=True).cuda(1)
res50 = torch.nn.Sequential(*list(res50.children())[:-1])
res50.eval()
preprocess = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])
FIELDNAMES = ['news_id', 'area', 'roi_location', 'features']
news_image_len=111313
roi_num=30
news_roi_feature = np.zeros((news_image_len,roi_num,2048),dtype='float32')
news_roi_location = np.zeros((news_image_len,roi_num,5),dtype='float32')
news_roi_mask = np.zeros((news_image_len,roi_num), dtype='int32')

with open(news_path,'r',encoding="UTF-8") as fin:
    cnt=1
    for line in fin:
        temp = line.split('\t')
        doc_id = temp[0]
        image = temp[5]
        if image != '':
            image=base64.b64decode(image)
            img_data = Image.open(BytesIO(image))
            im_resized_np = np.array(img_data)
            image_h = im_resized_np.shape[0]
            image_w = im_resized_np.shape[1]
            image_area = image_w*image_h
            if len(im_resized_np.shape)==2:
                img_data=img_data.convert('RGB')
                im_resized_np = np.array(img_data)
            if im_resized_np.shape[2]==4:
                im_resized_np=im_resized_np[:,:,:3]
            
            results = model.detect([im_resized_np])
            roi_batch = []
            result = results[0]
            rois = result['rois'].tolist()
            good_rois = []
            for roi in rois:
                good_rois.append(((roi[3]-roi[1])*(roi[2]-roi[0]),roi))
            good_rois = sorted(good_rois,key=lambda x :x[0],reverse=True)
            good_rois = good_rois[:roi_num]
            pic_roi_num = len(good_rois)
            for roi_idx,roi_t in enumerate(good_rois):
                area,roi = roi_t
                news_roi_location[cnt,roi_idx,0]=roi[0]/image_h
                news_roi_location[cnt,roi_idx,1]=roi[1]/image_w
                news_roi_location[cnt,roi_idx,2]=roi[2]/image_h
                news_roi_location[cnt,roi_idx,3]=roi[3]/image_w
                news_roi_location[cnt,roi_idx,4]=float(area)/image_area
                roi_image = Image.fromarray(im_resized_np[roi[0]:roi[2],roi[1]:roi[3],:]).convert('RGB')
                roi_batch.append(preprocess(roi_image))
            if pic_roi_num!=0:
                roi_batch = torch.stack(roi_batch,0).cuda(1)
                features = res50(roi_batch) 
                features = features.squeeze(-1).squeeze(-1).cpu().detach().tolist()
            
                news_roi_feature[cnt,:pic_roi_num]=features
                news_roi_mask[cnt,:pic_roi_num]=1
            

            cnt+=1

            
            if cnt%100==0:
                print(f"done {cnt}")
    print(f'{cnt}')
    np.savez(roi_file,features=news_roi_feature,location=news_roi_location,mask=news_roi_mask)

