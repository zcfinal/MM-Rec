import numpy as np
import torch
import pickle
import hashlib
import logging
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import os
import sys
import logging
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import Dataset, DataLoader
from preprocess import read_news_bert, get_doc_input_bert,read_news_image_size,read_news_image
from model import mmrec
from parameters import parse_args
import ipdb
import pickle
from transformers import AutoTokenizer
from model import BertConfig

#layer for finetuning
finetuneset={
    'news_encoder.bert.encoder.layer.7',
    'news_encoder.bert.encoder.layer.6',
    'news_encoder.bert.encoder.layer.5',
    'news_encoder.bert.encoder.layer.4',
    'news_encoder.bert.encoder.c_layer.0',
    'news_encoder.bert.encoder.c_layer.1',
    'news_encoder.bert.encoder.v_layer.0',
    'news_encoder.bert.encoder.v_layer.1',
    'news_encoder.bert.t_pooler',
    'news_encoder.bert.v_pooler'
}

def train(args):
    assert args.enable_hvd  # TODO
    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_json_file(args.config_file)

    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        os.path.join(args.root_data_dir,
                    f'subnews.tsv'), 
        args,
        tokenizer
    )
    
    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory= get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)
    
    image_size = read_news_image_size(args.image_size_file)
    news_imageid_dict,news_roi_feature,news_roi_location,news_roi_mask,news_image_whole =read_news_image(args.roi_file,args.whole_file,image_size,args)

    news_id2image_id = {}
    for news_id,image_id in news_imageid_dict.items():
        news_id2image_id[news_index[news_id]]=image_id

    model = mmrec(config,args)
    start_epoch=0
    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path,map_location='cpu')
        start_epoch = int(ckpt_path.split('-')[-1].split('.')[0])
        if hvd_rank == 0:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Model loaded from {ckpt_path}")
            logging.info(f"start from epoch [{start_epoch}]")
        del checkpoint
    
    model.train()

    for name,para in model.named_parameters():
        logging.info(name)
        req_grad = False
        for name_finetune in finetuneset:
            if name_finetune in name:
                req_grad = True
                break
        para.requires_grad = req_grad

    if args.enable_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Average)


    dataloader = DataLoaderTrain(
        news_index=news_index,
        news_combined=news_combined,
        data_dir=os.path.join(args.root_data_dir,
                            f'{args.dataset}/{args.train_dir}'),
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
        news_imageid_dict=news_id2image_id,
        news_roi_feature=news_roi_feature,
        news_roi_location=news_roi_location,
        news_roi_mask=news_roi_mask,
        news_image_whole=news_image_whole
    )
    logging.info('Training...')

    best_loss = 2
    args.max_steps_per_epoch = args.max_steps_per_epoch//(args.hvd_size*args.batch_size)

    for ep in range(start_epoch,args.epochs):
        loss = 0.0
        hvd.join()
        for cnt, (news_feature, log_ids, log_mask, input_ids, targets,input_imgs,image_loc,image_mask) in enumerate(dataloader,start=1):
            if cnt > args.max_steps_per_epoch or (args.debug and cnt>10):
                break

            if args.enable_gpu:
                news_feature = news_feature.cuda(non_blocking=True)
                log_ids = log_ids.cuda(non_blocking=True)
                log_mask = log_mask.cuda(non_blocking=True)
                input_ids = input_ids.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                input_imgs=input_imgs.cuda(non_blocking=True)
                image_loc=image_loc.cuda(non_blocking=True)
                image_mask = image_mask.cuda(non_blocking=True)

            feature_dict = {
                'input_txt':torch.narrow(news_feature,1,0,args.num_words_title),
                'input_imgs':input_imgs,
                'image_loc':image_loc,
                'token_type_ids': torch.narrow(news_feature,1,args.num_words_title,args.num_words_title),
                'attention_mask':torch.narrow(news_feature,1,args.num_words_title*2,args.num_words_title),
                'image_attention_mask':image_mask
            }

            bz_loss, y_hat = model(feature_dict, input_ids, log_ids, log_mask, targets)
            loss += bz_loss.data.float()
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            batch_loss = bz_loss.clone().detach()
            avg = hvd.allreduce(loss.clone().detach())
            avg_batch = hvd.allreduce(batch_loss)

            if ((cnt!=0 and cnt % args.log_steps == 0) or (cnt != 0 and args.debug)) and hvd_rank == 0:
                logging.info(
                    'epoch [{}] [{}] Ed: {}, train_avg_cumu_loss: {:.5f}, train_avg_batch_loss: {:.5f}'.format(
                        ep, hvd_rank, cnt * args.batch_size, avg.data/cnt, avg_batch.data ))

        # save model last of epoch
        if hvd_rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'category_dict': category_dict,
                    'domain_dict': domain_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")



def test(args):

    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)

    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path,map_location='cpu')

    if 'subcategory_dict' in checkpoint:
        subcategory_dict = checkpoint['subcategory_dict']
    else:
        subcategory_dict = {}

    category_dict = checkpoint['category_dict']
    domain_dict = checkpoint['domain_dict']

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_json_file(args.config_file)
    model = mmrec(config,args)
    
    if args.enable_gpu:
        model.cuda()
    if hvd_rank == 0:
        model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index = read_news_bert(
        os.path.join(args.root_data_dir,
                    f'subnews.tsv'), 
        args,
        tokenizer,
        'test'
    )
    word_dict = None

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory= get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)

    image_size = read_news_image_size(args.image_size_file)
    news_imageid_dict,news_roi_feature,news_roi_location,news_roi_mask,news_image_whole =read_news_image(args.roi_file,args.whole_file,image_size,args)

    news_id2image_id = {}
    for news_id,image_id in news_imageid_dict.items():
        news_id2image_id[news_index[news_id]]=image_id

    class NewsDataset(Dataset):
        def __init__(self, data,roi_feature,roi_location,roi_mask,image_whole,news_id2image_id):
            self.data = data
            self.roi_feature = roi_feature
            self.roi_location = roi_location
            self.roi_mask = roi_mask
            self.image_whole = image_whole
            self.news_id2image_id = news_id2image_id

        def __getitem__(self, idx):
            image_idx = self.news_id2image_id[idx] if idx in self.news_id2image_id else 0
            return self.data[idx],self.roi_feature[image_idx],self.roi_location[image_idx],self.roi_mask[image_idx],self.image_whole[image_idx]

        def __len__(self):
            return self.data.shape[0]

    def news_collate_fn(arr):
        mini_batch_news_feature,\
        mini_batch_news_roi_feature,\
        mini_batch_news_roi_location,\
        mini_batch_news_roi_mask,\
        mini_batch_news_image_whole =[],[],[],[],[]
        for news_feature,roi_feature,roi_location,roi_mask,image_whole in arr:
            mini_batch_news_feature.append(news_feature)
            mini_batch_news_roi_feature.append(roi_feature)
            mini_batch_news_roi_location.append(roi_location)
            mini_batch_news_roi_mask.append(roi_mask)
            mini_batch_news_image_whole.append(image_whole)
        mini_batch_news_feature = np.array(mini_batch_news_feature)
        mini_batch_news_roi_feature = np.array(mini_batch_news_roi_feature)
        mini_batch_news_roi_location = np.array(mini_batch_news_roi_location)
        mini_batch_news_roi_mask = np.array(mini_batch_news_roi_mask)
        mini_batch_news_image_whole = np.array(mini_batch_news_image_whole)

        batch_size = mini_batch_news_roi_location.shape[0]

        input_imgs = np.concatenate([mini_batch_news_image_whole,mini_batch_news_roi_feature],1)
        image_loc = np.concatenate([np.expand_dims(np.array([[0,0,1,1,1]],dtype=np.float32).repeat(batch_size,0),1),mini_batch_news_roi_location],1)
        image_mask = np.concatenate([np.array([[1]],dtype=np.float32).repeat(batch_size,0),mini_batch_news_roi_mask],1)

        news_feature= torch.LongTensor(mini_batch_news_feature)
        input_imgs = torch.FloatTensor(input_imgs)
        image_loc = torch.FloatTensor(image_loc)
        image_mask = torch.FloatTensor(image_mask)

        return news_feature,input_imgs,image_loc,image_mask

    news_dataset = NewsDataset(news_combined,news_roi_feature,news_roi_location,news_roi_mask,news_image_whole,news_id2image_id)
    news_dataloader = DataLoader(news_dataset,
                                batch_size=args.batch_size * 4,
                                num_workers=args.num_workers,
                                collate_fn=news_collate_fn)

    news_scoring_t = []
    news_scoring_v = []
    with torch.no_grad():
        for input_ids,input_imgs,image_loc,image_mask in tqdm(news_dataloader):
            input_ids = input_ids.cuda()
            feature_dict = {
                'input_txt':torch.narrow(input_ids,1,0,args.num_words_title),
                'input_imgs':input_imgs.cuda(),
                'image_loc':image_loc.cuda(),
                'token_type_ids': torch.narrow(input_ids,1,args.num_words_title,args.num_words_title),
                'attention_mask':torch.narrow(input_ids,1,args.num_words_title*2,args.num_words_title),
                'image_attention_mask':image_mask.cuda()
            }
            news_vec_t,news_vec_v = model.news_encoder(**feature_dict)
            news_vec_t = news_vec_t.to(torch.device("cpu")).detach().numpy()
            news_vec_v = news_vec_v.to(torch.device("cpu")).detach().numpy()
            news_scoring_t.extend(news_vec_t)
            news_scoring_v.extend(news_vec_v)


    news_scoring_t = np.array(news_scoring_t)
    news_scoring_v = np.array(news_scoring_v)

    logging.info("news scoring num: {}".format(news_scoring_t.shape[0]))
 

    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring_t=news_scoring_t,
        news_scoring_v=news_scoring_v,
        word_dict=word_dict,
        news_bias_scoring= None,
        data_dir=os.path.join(args.root_data_dir,
                            f'{args.dataset}/{args.test_dir}'),
        filename_pat='test_*.tsv',
        args=args,
        world_size=hvd_size,
        news_id2image_id = news_id2image_id,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )

    from metrics import roc_auc_score, ndcg_score, mrr_score, ctr_score

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
            '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    for cnt, (log_vecs_t,log_vecs_v, log_masks, news_vecs_t,news_vecs_v, news_bias, labels) in enumerate(dataloader):

        for index, user_vec_t,user_vec_v, news_vec_t,news_vec_v, label, log_mask in zip(
                range(len(labels)), log_vecs_t,log_vecs_v, news_vecs_t,news_vecs_v, labels,log_masks):
                
            if label.mean() == 0 or label.mean() == 1:
                continue

            if args.enable_gpu:
                user_vec_t = torch.FloatTensor(user_vec_t).cuda(non_blocking=True).unsqueeze(0)
                user_vec_v = torch.FloatTensor(user_vec_v).cuda(non_blocking=True).unsqueeze(0)
                news_vec_t = torch.FloatTensor(news_vec_t).cuda(non_blocking=True).unsqueeze(0)
                news_vec_v = torch.FloatTensor(news_vec_v).cuda(non_blocking=True).unsqueeze(0)
                log_mask = torch.FloatTensor(log_mask).cuda(non_blocking=True).unsqueeze(0)

            user_vecs = model.user_encoder(news_vec_t,news_vec_v,user_vec_t,user_vec_v, log_mask)

            score = (news_vec_t+news_vec_v)*user_vecs
            score = torch.sum(score,-1)
            score = score.squeeze(0).cpu().detach().numpy()

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        if cnt % args.log_steps == 0:

            print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]))
    
    with open(os.path.join(args.log_dir,f'test_result_{hvd_rank}.txt'),'w') as fout :
        fout.write(str(np.mean(AUC))+' '+str(len(AUC))+'\n')
        fout.write(str(np.mean(MRR))+' '+str(len(MRR))+'\n')
        fout.write(str(np.mean(nDCG5))+' '+str(len(nDCG5))+'\n')
        fout.write(str(np.mean(nDCG10))+' '+str(len(nDCG10))+'\n')

    # stop scoring
    dataloader.join()

if __name__ == "__main__":
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if 'cal' in args.mode:
        metric = [[0,0] for i in range(4)]
        for i in range(args.hvd_size):
            with open(os.path.join(args.log_dir,f'test_result_{i}.txt'),'r')as f:
                cnt = 0
                for line in f:
                    temp = line.split(' ')
                    avg = float(temp[0])
                    tot = float(temp[1])
                    sum_val = avg*tot
                    metric[cnt][0]+=sum_val
                    metric[cnt][1]+=tot
                    cnt+=1
        with open(os.path.join(args.log_dir,'final_result.txt'),'w') as fout :
            for i in range(4):
                fout.write(str(metric[i][0]/metric[i][1])+' '+str(metric[i][1])+'\n')
        exit()
    if 'train' in args.mode:
        utils.setuplogger(os.path.join(args.log_dir,'log_train.txt'))
        logging.info(args)
        train(args)
    if 'test' in args.mode:
        utils.setuplogger(os.path.join(args.log_dir,'log_test.txt'))
        logging.info(args)
        test(args)


