import sys
import traceback
import logging
import time
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset
from streaming import StreamSampler, StreamSamplerTest
import utils
import ipdb

def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


class DataLoaderTrain(IterableDataset):
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_combined,
                 news_imageid_dict,
                 news_roi_feature,
                 news_roi_location,
                 news_roi_mask,
                 news_image_whole,
                 word_dict = None,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.shuffle_buffer_size = args.shuffle_buffer_size

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_combined = news_combined
        self.news_index = news_index
        self.word_dict = word_dict
        self.news_imageid_dict=news_imageid_dict
        self.news_roi_feature=news_roi_feature
        self.news_roi_location=news_roi_location
        self.news_roi_mask=news_roi_mask
        self.news_image_whole=news_image_whole

    def start(self):
        self.epoch += 1
        self.sampler = StreamSampler(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def trans_to_imageindex(self, nids):
        return [self.news_imageid_dict[i] if i in self.news_imageid_dict else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value]*(fix_length-len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (len(x) - fix_length)
        return pad_x, mask

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSampler(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
            self.outputs.put(None)
            self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise
    
    def newsample(self,news,ratio):
        if ratio >len(news):
            return news + [0]*(ratio-len(news))
        else:
            return random.sample(news,ratio)

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def parse_sent(self, sent, fix_length):
        sent = [self.word_dict[w] if w in self.word_dict else 0 for w in utils.word_tokenize(sent)]
        sent, _ = self.pad_to_fix_len(sent, fix_length, padding_front=False)
        return sent

    def parse_sents(self, sents, max_sents_num, max_sent_length, padding_front=True):
        sents, sents_mask = self.pad_to_fix_len(sents, max_sents_num, padding_value='')
        sents = [self.parse_sent(s, max_sent_length) for s in sents]
        sents = np.stack(sents, axis=0)
        sents_mask = np.array(sents_mask)
        return sents, sents_mask

    def _process(self, batch):
        user_feature_batch, log_mask_batch, news_feature_batch, label_batch = [], [], [], []
        news_batch = []

        for line in batch:
            line = line.decode()
            splited = line.replace('\n','').split('\t')
            history = splited[3].split(' ')[-self.user_log_length:]
            poss = splited[6].split(' ')
            neg = splited[7].split(' ')

            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(history),self.user_log_length)

            news_batch.extend(click_docs)

            for pdoc in poss:
                negps=self.newsample(neg,self.npratio)
                sample_news = [pdoc] + negps
                sample_news = self.trans_to_nindex(sample_news)
                label = 0
                
                user_feature_batch.append(click_docs)
                log_mask_batch.append(log_mask)
                news_feature_batch.append(sample_news)
                label_batch.append(label)

                news_batch.extend(sample_news)
        
        news_batch = list(set(news_batch))
        news_image_batch = self.trans_to_imageindex(news_batch)
        mini_batch_news_feature = self.news_combined[news_batch]
        mini_batch_news_roi_feature = self.news_roi_feature[news_image_batch]
        mini_batch_news_roi_location = self.news_roi_location[news_image_batch]
        mini_batch_news_roi_mask = self.news_roi_mask[news_image_batch]
        mini_batch_news_image_whole = self.news_image_whole[news_image_batch]

        batch_size = mini_batch_news_roi_location.shape[0]
        input_imgs = np.concatenate([mini_batch_news_image_whole,mini_batch_news_roi_feature],1)
        image_loc = np.concatenate([np.expand_dims(np.array([[0,0,1,1,1]],dtype=np.float32).repeat(batch_size,0),1),mini_batch_news_roi_location],1)
        image_mask = np.concatenate([np.array([[1]],dtype=np.float32).repeat(batch_size,0),mini_batch_news_roi_mask],1)

        newsid2location = {news:loc for loc,news in enumerate(news_batch)}

        user_feature_batch_loc = [[newsid2location[doc] for doc in click_docs] for click_docs in user_feature_batch]
        news_feature_batch_loc = [[newsid2location[doc] for doc in click_docs] for click_docs in news_feature_batch]

        if self.enable_gpu:
            mini_batch_news_feature = torch.LongTensor(mini_batch_news_feature).cuda()
            user_feature_batch = torch.LongTensor(user_feature_batch_loc).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            news_feature_batch = torch.LongTensor(news_feature_batch_loc).cuda()
            label_batch = torch.LongTensor(label_batch).cuda()
            input_imgs = torch.FloatTensor(input_imgs).cuda()
            image_loc = torch.FloatTensor(image_loc).cuda()
            image_mask = torch.FloatTensor(image_mask).cuda()
        else:
            mini_batch_news_feature = torch.LongTensor(mini_batch_news_feature)
            user_feature_batch = torch.LongTensor(user_feature_batch_loc)
            log_mask_batch = torch.FloatTensor(log_mask_batch)
            news_feature_batch = torch.LongTensor(news_feature_batch_loc)
            label_batch = torch.LongTensor(label_batch)
            input_imgs = torch.FloatTensor(input_imgs).cuda()
            image_loc = torch.FloatTensor(image_loc).cuda()
            image_mask = torch.FloatTensor(image_mask).cuda()

        return mini_batch_news_feature,user_feature_batch, log_mask_batch, news_feature_batch, label_batch, \
        input_imgs,image_loc,image_mask

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration
        if self.enable_prefetch:
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
        else:
            next_batch = self._process(self.sampler.__next__())
            return next_batch

        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration

        return next_batch

    def join(self):
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:#if data queue has processed data
                    self.outputs.get()#get data
                    self.outputs.task_done()#tell the queue 
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None


class DataLoaderTest(DataLoaderTrain):
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_scoring_t,
                 news_scoring_v,
                 news_id2image_id,
                 word_dict=None,
                 news_bias_scoring=None,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_scoring_t = news_scoring_t
        self.news_scoring_v = news_scoring_v
        self.news_bias_scoring = news_bias_scoring
        self.news_index = news_index
        self.word_dict = word_dict
        self.news_id2image_id = news_id2image_id

    def start(self):
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSamplerTest(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
            self.outputs.put(None)
            self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _process(self, batch):
        user_feature_t_batch,user_feature_v_batch, log_mask_batch, news_feature_t_batch,news_feature_v_batch, news_bias_batch, label_batch =[] ,[], [], [], [], [],[]

        for line in batch:
            line = line.decode()
            splited = line.replace('\n','').split('\t')
            history = splited[3].split(' ')[-self.user_log_length:]
            poss = splited[6].split(' ')
            neg = splited[7].split(' ')

            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(history),self.user_log_length)

            click_docs_vecs_t = self.news_scoring_t[click_docs]
            click_docs_vecs_v = self.news_scoring_v[click_docs]

            poss = self.trans_to_nindex(poss)
            neg = self.trans_to_nindex(neg)
            imp_docs = poss+neg
            imp_docs_vecs_t = self.news_scoring_t[imp_docs]
            imp_docs_vecs_v = self.news_scoring_v[imp_docs]
            labels = [1]*len(poss) + [0]*len(neg)

            user_feature_t_batch.append(click_docs_vecs_t)
            user_feature_v_batch.append(click_docs_vecs_v)
            news_feature_t_batch.append(imp_docs_vecs_t)
            news_feature_v_batch.append(imp_docs_vecs_v)
            label_batch.append(np.array(labels))
            log_mask_batch.append(log_mask)

        return user_feature_t_batch, user_feature_v_batch, log_mask_batch, news_feature_t_batch , news_feature_v_batch, news_bias_batch, label_batch





