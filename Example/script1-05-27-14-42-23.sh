#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! our NF1 data folder 
our_data_folder=/data/duongdb/DeployOnline/ClassifyNF1/Example # ! change this on your machine

# ! model hyperparams
weight=10
learningrate=1e-05
imagesize=448
schedulerscaler=10 
dropout=0.2

# ! make sure to call correct crop (or focused images) vs. recrop (or panoramic images)
imagetype='recrop'

batchsize=64 # may want to lower this to fit in smaller GPU
ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=9c_b4ns_$imagesize'_ext_15ep-newfold' # ! this says to use "b4ns" efficient net b4 noisy student. 

model_folder_name=b4ns$imagesize$imagetype'Wl'$weight'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest

our_data_dir=$our_data_folder/Crop
if [ $imagetype = 'recrop' ]
then
  our_data_dir=$our_data_folder/Recrop
fi
our_train_csv=$our_data_dir/train.csv # ! download this train.csv from our github

# ! data dir of SIIM datasets

datadir=/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte # ! change this on your machine
modeldir=$our_data_folder/$model_folder_name 
mkdir $modeldir

logdir=$our_data_folder/$model_folder_name 
oofdir=$our_data_folder/$model_folder_name/EvalDev 

# ! now we train/test

cd /data/duongdb/DeployOnline/ClassifyNF1 # ! change this on your machine

# ! train
# python train.py --kernel-type $kernel_type --data-dir $datadir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir --num-workers 8 --fold '1' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --weighted-loss $weight --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest


# ! eval on left-out fold
# python evaluate.py --kernel-type $kernel_type --data-dir $datadir --model-dir $modeldir --log-dir $logdir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 8 --fold '1' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --n-test $ntest


# ! eval on test set (test set is coded as fold=5)
python evaluate.py --kernel-type $kernel_type --data-dir $datadir --model-dir $modeldir --log-dir $logdir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 8 --fold '1' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest


# ! look at pixels in test set. 
# for condition in MA ML HMI IP NF1 TSC EverythingElse
# do
# python evaluate.py --kernel-type $kernel_type --data-dir $datadir --model-dir $modeldir --log-dir $logdir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 1 --num-workers 8 --fold '1' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --dropout $dropout --do_test --n-test $ntest --attribution_keyword $condition --outlier_perc 1
# done

