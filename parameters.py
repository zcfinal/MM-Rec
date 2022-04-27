import argparse
import utils
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        choices=['train', 'test', 'train_test','cal'])
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default=
        "./data",
    )
    parser.add_argument("--hvd_size",type=int,default=1)
    parser.add_argument("--valid_dir",type=str,default=None)
    parser.add_argument("--dataset",
                        type=str,
                        default='MIND')
    parser.add_argument(
        "--from_pretrained",
        default="../../model/mmrec/pytorch_model_8.bin",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument("--config_file",type=str,default='../../model/mmrec/config.json')
    parser.add_argument("--exp_name",type=str,default="default")
    parser.add_argument("--log_dir",type=str,default="../../log")
    parser.add_argument(
        "--train_dir",
        type=str,
        default='train',
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default='test',
    )
    parser.add_argument('--roi_npz_file',type=str,default='../../data/rois.npz',
                        help='cached file stores roi information')
    parser.add_argument("--whole_file",type=str,default='../../data/whole_features.tsv',
                        help='original file stores whole features')
    parser.add_argument("--roi_file",type=str,default='../../data/rois_np',
                        help='original file stores roi features')
    parser.add_argument("--image_size_file",type=str,default='../../data/image_size.tsv',
                        help='original file stores image size')
    parser.add_argument("--filename_pat", type=str, default="train_*.tsv")
    parser.add_argument("--model_dir", type=str, default='./model')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--npratio", type=int, default=1)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--enable_hvd", type=utils.str2bool, default=True)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--filter_num", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=500)

    # model training
    parser.add_argument("--debug", type=utils.str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--news_attributes",
        type=str,
        nargs='+',
        default=['title','category','subcategory'],
        choices=['title', 'abstract', 'body', 'category', 'domain', 'subcategory', 'image'])
       
    parser.add_argument("--num_words_title", type=int, default=24)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--num_words_uet", type=int, default=16)
    parser.add_argument("--num_words_bing", type=int, default=26)
    parser.add_argument(
        "--user_log_length",
        type=int,
        default=50,
    )

    parser.add_argument("--log_file",type=str,default=None)
    parser.add_argument("--save_steps", type=int, default=1000000000)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default=None,
        help="choose which ckpt to load and test"
    )
    args = parser.parse_args()

    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
