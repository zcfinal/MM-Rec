# Source codes for paper "MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation".

1. To get the image rois, you may need to use the following file:

./get_roi/Mask_RCNN/samples/get_image_size.py is the file to get news image size.

./get_roi/Mask_RCNN/samples/get_roi.py is the file to get the news image rois.

./get_roi/Mask_RCNN/samples/get_whole_feature.py is the file to get the features of the overall news images.

2. For the pretrained news encoder, you can go to https://github.com/jiasenlu/vilbert_beta to download it.

3. This pipeline is based on the horovod framework. We run experiments on 8 GPUs. You need to split the original behaviors file in the news dataset into X parts (named as train_x.tsv, X∈ {0, 1, 2, ..., X}) if you use X GPUs to train.

Use run.sh to start the program.
