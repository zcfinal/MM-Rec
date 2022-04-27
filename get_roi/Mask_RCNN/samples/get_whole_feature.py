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

news_path = "/workspaceblobstore/data/v-chaozhang/mmrec/data/subnews.tsv"
whole_file = "/workspaceblobstore/data/v-chaozhang/mmrec/data/whole_features.tsv"

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
FIELDNAMES = ['news_id', 'features']

news_whole_feature = np.zeros((111313,1,2048),dtype='float32')
with open(news_path,'r',encoding="UTF-8") as fin,open(whole_file,'w')as tsvfile:
    writer = csv.DictWriter(tsvfile, delimiter = '\t',fieldnames = FIELDNAMES)  
    cnt=0
    news_id_batch = []
    feature_batch = []
    for line in fin:
        temp = line.split('\t')
        doc_id = temp[0]
        image = temp[5]
        if image != '':
            image=base64.b64decode(image)
            img_data = Image.open(BytesIO(image)).convert('RGB')
            news_id_batch.append(doc_id)
            feature_batch.append(preprocess(img_data))
            if len(news_id_batch)==32:
                feature_batch = torch.stack(feature_batch,0).cuda(1)
                features = res50(feature_batch) 
                features = features.squeeze(-1).squeeze(-1).cpu().detach().tolist()
                for doc_id,feature in zip(news_id_batch,features):
                    news_dict = {
                    'news_id':doc_id,
                    'features':feature
                    }
                    writer.writerow(news_dict)
                    cnt+=1
                news_id_batch,feature_batch=[],[]
                if cnt%3200==0:
                    print(f"done {cnt}")
    if len(news_id_batch)!=0:
        feature_batch = torch.stack(feature_batch,0).cuda(1)
        features = res50(feature_batch) 
        features = features.squeeze(-1).squeeze(-1).cpu().detach().tolist()
        for doc_id,feature in zip(news_id_batch,features):
            news_dict = {
            'news_id':doc_id,
            'features':feature
            }
            writer.writerow(news_dict)
            cnt+=1
    print(cnt)