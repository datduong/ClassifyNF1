source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyNF1

for modelname in b4ns448cropWl5ss10lr3e-05dp0.2b64ntest1 b4ns448recropWl10ss10lr1e-05dp0.2b64ntest1 
do
cd /data/duongdb/ClassifyNF1
modeldir="/data/duongdb/DeployOnline/ClassifyNF1/Example/"$modelname
labels="AK,BCC,BKL,DF,EverythingElse,HMI,IP,MA,ML,NF1,SCC,TSC,VASC,melanoma,nevus,unknown"
python3 ensemble_our_classifier.py --model-dir $modeldir --labels $labels --NF1 1
done 

