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
whole_file = "/workspaceblobstore/data/v-chaozhang/mmrec/data/image_size.tsv"


FIELDNAMES = ['news_id', 'image_w','image_h']
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
            img_data = np.array(img_data)
            news_dict = {
            'news_id':doc_id,
            'image_w':img_data.shape[1],
            'image_h':img_data.shape[0]
            }
            writer.writerow(news_dict)
