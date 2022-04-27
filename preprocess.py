from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
from utils import word_tokenize
import base64
from io import BytesIO
from PIL import Image
import csv
import ipdb
import ast
import os
import pickle

def get_domain(url):
    domain = urlparse(url).netloc
    return domain

#get news infomation
def read_news_bert(news_path, args, tokenizer, mode='train'):
    news = {}
    categories = []
    subcategories = []
    domains = []
    news_index = {}
    index = 1
    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            if args.debug:
                if index==100:
                    break
            splited = line.replace('#N#',' ').replace('#R#',' ').replace('#TAB#',' ').replace('\n','').split('\t')
            
            doc_id = splited[0]
            category = splited[1]
            subcategory = splited[2]
            title = splited[3]

            news_index[doc_id] = index
            index += 1

            if 'title' in args.news_attributes:
                title = title.lower()
                title = tokenizer(title, max_length=args.num_words_title, \
                pad_to_max_length=True, truncation=True)
            else:
                title = []

            if 'abstract' in args.news_attributes:
                abstract = abstract.lower()
                abstract = tokenizer(abstract, max_length=args.num_words_abstract, \
                pad_to_max_length=True, truncation=True)
            else:
                abstract = []

            if 'body' in args.news_attributes:
                body = body.lower()[:2000]
                body = tokenizer(body, max_length=args.num_words_body, \
                pad_to_max_length=True, truncation=True)
            else:
                body = []

            if 'category' in args.news_attributes:
                categories.append(category)
            else:
                category = None
            
            if 'subcategory' in args.news_attributes:
                subcategories.append(subcategory)
            else:
                subcategory = None

            if 'domain' in args.news_attributes:
                domain = get_domain(url)
                domains.append(domain)
            else:
                domain = None


            news[doc_id] = [category, subcategory, title]

    if mode == 'train':
        categories = list(set(categories))
        category_dict = {}
        index = 1
        for x in categories:
            category_dict[x] = index
            index += 1

        subcategories = list(set(subcategories))
        subcategory_dict = {}
        index = 1
        for x in subcategories:
            subcategory_dict[x] = index
            index += 1

        domains = list(set(domains))
        domain_dict = {}
        index = 1
        for x in domains:
            domain_dict[x] = index
            index += 1

        return news, news_index, category_dict, domain_dict, subcategory_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'

#tokenize
def get_doc_input_bert(news, news_index, category_dict, domain_dict, subcategory_dict, args):
    news_num = len(news) + 1
    if 'title' in args.news_attributes:
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_type = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_attmask = np.zeros((news_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_type = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((news_num, args.num_words_abstract), dtype='int32')
        news_abstract_type = np.zeros((news_num, args.num_words_abstract), dtype='int32')
        news_abstract_attmask = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_type = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_type = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((news_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_type = None
        news_body_attmask = None
    if 'category' in args.news_attributes:
        news_category = np.zeros((news_num, 1), dtype='int32')
    else:
        news_category = None

    if 'domain' in args.news_attributes:
        news_domain = np.zeros((news_num, 1), dtype='int32')
    else:
        news_domain = None

    if 'subcategory' in args.news_attributes:
        news_subcategory = np.zeros((news_num, 1), dtype='int32')
    else:
        news_subcategory = None


    for key in tqdm(news):
        category, subcategory, title= news[key]
        doc_index = news_index[key]

        if 'title' in args.news_attributes:
            news_title[doc_index] = title['input_ids']
            news_title_type[doc_index] = title['token_type_ids']
            news_title_attmask[doc_index] = title['attention_mask']          

        if 'abstract' in args.news_attributes:
            news_abstract[doc_index] = abstract['input_ids']
            news_abstract_type[doc_index] = abstract['token_type_ids']
            news_abstract_attmask[doc_index] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[doc_index] = body['input_ids']
            news_body_type[doc_index] = body['token_type_ids']
            news_body_attmask[doc_index] = body['attention_mask']

        if 'category' in args.news_attributes:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        
        if 'subcategory' in args.news_attributes:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
        
        if 'domain' in args.news_attributes:
            news_domain[doc_index, 0] = domain_dict[domain] if domain in domain_dict else 0

    return news_title, news_title_type, news_title_attmask, \
           news_abstract, news_abstract_type, news_abstract_attmask, \
           news_body, news_body_type, news_body_attmask, \
           news_category, news_domain, news_subcategory

def read_news_image_size(path):
    image_size = {}
    FIELDNAMES = ['news_id', 'image_w','image_h']
    with open(path,'r')as fin:
        reader = csv.DictReader(fin, delimiter='\t',fieldnames = FIELDNAMES)
        for item in reader:
            image_size[item['news_id']]=(float(item['image_h']),float(item['image_w']))
            
    return image_size

def read_news_image(path_roi,path_whole,image_size,args):
    csv.field_size_limit(2048*1000)
    news_imageid_dict={name:idx for idx,name in enumerate(image_size,start=1)}
    news_image_len = len(image_size)+1
    roi_num=30
    news_roi_feature = np.zeros((news_image_len,roi_num,2048),dtype='float32')
    news_roi_location = np.zeros((news_image_len,roi_num,5),dtype='float32')
    news_roi_mask = np.zeros((news_image_len,roi_num), dtype='int32')
    news_whole_feature = np.zeros((news_image_len,1,2048),dtype='float32')
    # cached roi data
    if os.path.exists(args.roi_npz_file):
        roi = np.load(args.roi_npz_file)
        news_roi_feature = roi['features']
        news_roi_location = roi['location']
        news_roi_mask = roi['mask']
        news_whole_feature = roi['whole']
        return news_imageid_dict,news_roi_feature,news_roi_location,news_roi_mask,news_whole_feature

    # not cached
    if os.path.exists(args.roi_file):
        roi = np.load(args.roi_file)
        news_roi_feature = roi['features']
        news_roi_location = roi['location']
        news_roi_mask = roi['mask']

    FIELDNAMES = ['news_id', 'features']
    with open(path_whole,'r') as fin:
        cnt_line = 1
        reader = csv.DictReader(fin, delimiter='\t',fieldnames = FIELDNAMES)
        for item in reader:
            cnt = news_imageid_dict[item['news_id']]
            news_whole_feature[cnt] = np.array(ast.literal_eval(item['features'])).reshape((1,2048))
            cnt_line+=1
            if args.debug and cnt_line==100:
                break
            if cnt_line%100==0:
                print(f'image whole read {cnt_line}')
        
    np.savez(args.roi_npz_file,features=news_roi_feature,location=news_roi_location,mask=news_roi_mask,whole=news_whole_feature)
    return news_imageid_dict,news_roi_feature,news_roi_location,news_roi_mask,news_whole_feature




if __name__ == "__main__":

    from parameters import parse_args
    args = parse_args()
    image_size = read_news_image_size(args.image_size_file)
    read_news_image(args.roi_file,args.whole_file,image_size,args)


    