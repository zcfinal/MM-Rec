hvd_size=8
mode=$1 # ['train', 'test', 'train_test', 'cal']
model_dir=$3 
log_dir=$4
load_ckpt_name=$5
exp_name=$6
news_attributes=$2
echo ${news_attributes}
root_data_dir=../../data
train_dir="train"
valid_dir="validation"
test_dir="test"
dataset="MSNImage"

epoch=3

batch_size=8

lr=0.00001
max_steps_per_epoch=1000000 
npratio=4
debug=False


if [ ${mode} == train ] 
then
    mpirun -np ${hvd_size} -H localhost:${hvd_size} \
    python run.py --root_data_dir ${root_data_dir} \
    --mode ${mode} --epoch ${epoch} --dataset ${dataset} \
    --model_dir ${model_dir}  --batch_size ${batch_size} \
    --news_attributes ${news_attributes} --lr ${lr} \
    --train_dir ${train_dir} --test_dir ${test_dir} \
    --max_steps_per_epoch ${max_steps_per_epoch} --debug ${debug} --exp_name ${exp_name} --hvd_size ${hvd_size} \
    --npratio ${npratio} --log_dir ${log_dir} --valid_dir ${valid_dir} \

elif [ ${mode} == test ]
then
    batch_size=16
    log_steps=100
    mpirun -np ${hvd_size} -H localhost:${hvd_size} \
    python run.py --root_data_dir ${root_data_dir} \
    --mode ${mode} --epoch ${epoch} --dataset ${dataset} \
    --model_dir ${model_dir}  --batch_size ${batch_size} \
    --news_attributes ${news_attributes} --lr ${lr} \
    --train_dir ${train_dir} --test_dir ${test_dir} \
    --log_steps ${log_steps} \
    --load_ckpt_name ${load_ckpt_name} --log_dir ${log_dir} --exp_name ${exp_name} --hvd_size ${hvd_size}
elif [ ${mode} == cal ]
then
    python run.py --mode ${mode} --exp_name ${exp_name} --log_dir ${log_dir} --hvd_size ${hvd_size}
fi