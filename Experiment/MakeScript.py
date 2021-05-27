import re,sys,os,pickle
from datetime import datetime

# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=24
# sbatch --partition=gpu --time=4:00:00 --gres=gpu:p100:1 --mem=16g --cpus-per-task=24
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=10g --cpus-per-task=20
# sbatch --time=12:00:00 --mem=100g --cpus-per-task=24

script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! our NF1 data folder 
our_data_folder=OURDATAFOLDER # ! change this on your machine

# ! model hyperparams
weight=WEIGHT
learningrate=LEARNRATE
imagesize=IMAGESIZE
schedulerscaler=ScheduleScaler 
dropout=DROPOUT

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
python train.py --kernel-type $kernel_type --data-dir $datadir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir --num-workers 8 --fold 'FOLD' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --weighted-loss $weight --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest


# ! eval on left-out fold
python evaluate.py --kernel-type $kernel_type --data-dir $datadir --model-dir $modeldir --log-dir $logdir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 8 --fold 'FOLD' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --n-test $ntest


# ! eval on test set (test set is coded as fold=5)
python evaluate.py --kernel-type $kernel_type --data-dir $datadir --model-dir $modeldir --log-dir $logdir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 64 --num-workers 8 --fold 'FOLD' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest


# ! look at pixels in test set. 
for condition in MA ML HMI IP NF1 TSC EverythingElse
do
python evaluate.py --kernel-type $kernel_type --data-dir $datadir --model-dir $modeldir --log-dir $logdir --data-folder 512 --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 1 --num-workers 8 --fold 'FOLD' --our-csv $our_train_csv --our-data-dir $our_data_dir --out-dim 16 --dropout $dropout --do_test --n-test $ntest --attribution_keyword $condition --outlier_perc 1
done

"""

workpath = '/data/duongdb/DeployOnline/ClassifyNF1/Example' # ! change this on your machine

os.chdir(workpath)

# b4ns448cropWl5ss10lr3e-05dp0.2 # ! best

# b4ns448recropWl10ss10lr1e-05dp0.2 # ! best

counter=0
for fold in [0,1,2,3,4]: 
  for imagesize in [448]: 
    for weight in [10]: # 5,10, # ! weighted loss
      for schedulerscaler in [10]: # ! hyper params
        for learn_rate in [0.00001]: # 0.00001,0.00003
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            script2 = re.sub('OURDATAFOLDER',str(workpath),script2)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('LEARNRATE',str(learn_rate),script2)
            script2 = re.sub('ScheduleScaler',str(schedulerscaler),script2)
            script2 = re.sub('FOLD',str(fold),script2)
            script2 = re.sub('DROPOUT',str(dropout),script2)
            now = datetime.now() # current date and time
            scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
            fout = open(scriptname,'w')
            fout.write(script2)
            fout.close()
            # 
            print ('create script {} at {}'.format(scriptname,workpath))
            # os.system('sbatch --partition=gpu --time=16:00:00 --gres=gpu:v100x:1 --mem=12g --cpus-per-task=16 ' + scriptname )
            # os.system('sbatch --partition=gpu --time=2:00:00 --gres=gpu:p100:1 --mem=20g --cpus-per-task=32 ' + scriptname )
            # os.system('sbatch --time=3:00:00 --mem=80g --cpus-per-task=20 ' + scriptname )
            counter = counter + 1 

#
exit()



# ! power
# from scipy.stats import norm
# import numpy as np
# percent_difference = 10
# standard_dev = 15
# people = 30
# 1 - norm.cdf ( 1.96, percent_difference/(standard_dev/np.sqrt(people)), 1) 
