exp_name=mmrec_8layer
news_attributes='title'
model_dir=../../model/${exp_name}
log_dir=../../log/${exp_name}
ckpt=epoch-1.pt
./demo.sh train ${news_attributes}  ${model_dir} ${log_dir} ${ckpt} ${exp_name}
sleep 10
./demo.sh test ${news_attributes}  ${model_dir} ${log_dir} ${ckpt} ${exp_name}
sleep 10
./demo.sh cal ${news_attributes}  ${model_dir} ${log_dir} ${ckpt} ${exp_name}