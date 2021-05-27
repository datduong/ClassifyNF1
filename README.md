## NF1 dataset 


**[Our current paper.](https://www.medrxiv.org/content/10.1101/2021.04.08.21255123v1)**


**[What is NF1?](https://www.cancer.net/cancer-types/neurofibromatosis-type-1)**


**[See some examples of NF1 images.](https://dermnetnz.org/topics/neurofibromatosis-images/)**


**_[Click to see an example of our survey to pediatricians and geneticists.](https://ncidccpssurveys.gov1.qualtrics.com/jfe/form/SV_2icqumxXrn2x2iG)_**


Please contact us for the data. 


[Pytorch pre-trained models (ends with .pth) are on this google drive here](https://drive.google.com/drive/folders/1m2c7uWPOkIK_FU3gTIpjJbqfpYHqE_0_?usp=sharing). 


## Instruction to train. 

Because our NF1 dataset is small, we need to borrow the power of a larger auxilary dataset. Here, we use the SIIM-ISIC dataset. You will need to download this dataset first.

### Download SIIM-ISIC dataset (you need to install [Kaggle API](https://github.com/Kaggle/kaggle-api))

Download the 2020 and 2019 data (which already had 2018 data) by Chris Deotte. You can choose to save at any location, below, I am using my own folder name `/data/duongdb/ISIC2020-SkinCancerBinary/`, you should change this folder name. 

```
mkdir /data/duongdb/ISIC2020-SkinCancerBinary/ # You can choose your own location.
cd /data/duongdb/ISIC2020-SkinCancerBinary/
for input_size in 512 
do
  kaggle datasets download -d cdeotte/jpeg-isic2019-${input_size}x${input_size}
  kaggle datasets download -d cdeotte/jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-melanoma-${input_size}x${input_size}.zip -d jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-isic2019-${input_size}x${input_size}.zip -d jpeg-isic2019-${input_size}x${input_size}
  rm jpeg-melanoma-${input_size}x${input_size}.zip jpeg-isic2019-${input_size}x${input_size}.zip
done
```

Detail at _(but not necessary so important to read)_ https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412. We care about the jpeg images, and not the tensorflow tfrecord, we will use just the images at size 512 x 512. 

## Train, test, see pixel attribution

Once you download the SIIM-ISIC dataset, you can download our NF1 dataset. These 2 datasets do not need to be in the same folder. 
  - I have SIIM-ISIC dataset in the location `/data/duongdb/ISIC2020-SkinCancerBinary/` and my NF1 dataset in the folder `/data/duongdb/DeployOnline/ClassifyNF1/Example/`.

`dataset.py` will read and format the images. The `train.py` and `evaluate.py` will call `dataset.py`, so you do not need to change `dataset.py`. 

  - For the SIIM-ISIC dataset, `dataset.py` will auto-detect these images, you will not need to do anything except for downloading the SIIM-ISIC data as explained above.

  - For our NF1 dataset, `dataset.py` requires a manually created `train.csv` ([we provide them here](https://github.com/datduong/ClassifyNF1/tree/master/TrainTestCsv) and the same files are also in [here](https://github.com/datduong/ClassifyNF1/tree/master/Example)) where each row contains the path to each image. Notice, we _do not_ need a `test.csv`. The `train.csv` contains 6 folds (starting from 0 to 5), the last fold (number 5) contains the test images. 

    - We provide two different `train.csv`, one for the _focused images_ (where just the skin condition is showing) and one for the _panoramic images_ (where other body parts, like arms and leg, may show). 

      - We call the _focused images_ as `Crop`, and _panoramic images_ as `Recrop`. I'm sorry for this confusion. 

      - The video below show how your NF1 data source needs to be. 

      [![asciicast](https://asciinema.org/a/B2T6TqERamgtm7JVScaIXmVcu.svg)](https://asciinema.org/a/B2T6TqERamgtm7JVScaIXmVcu)

Now, you **_train the model with the script in `Experiment/MakeScript.py`_**. 

  - `Experiment/MakeScript.py` creates shell scripts; for example `script1.sh`. To run this shell script, you can use `bash script1.sh`. Please note, we trained on Nvidia v100x, so your machine must have a GPU and pytorch properly installed. 

  - Please note, for me, the important folder paths are `/data/duongdb/ISIC2020-SkinCancerBinary/data-by-cdeotte` and `/data/duongdb/DeployOnline/ClassifyNF1/Example`. 

    - Please change these path names in `Experiment/MakeScript.py` according to your machine. You will need to change path names in just `Experiment/MakeScript.py` [here](https://github.com/datduong/ClassifyNF1/blob/master/Experiment/MakeScript.py#L46), [here](https://github.com/datduong/ClassifyNF1/blob/master/Experiment/MakeScript.py#L55) and [here](https://github.com/datduong/ClassifyNF1/blob/master/Experiment/MakeScript.py#L77)

    - To run _focused images_ [change `imagetype='crop'` in this line](https://github.com/datduong/ClassifyNF1/blob/master/Experiment/MakeScript.py#L28).

    - To run _panoramic images_ [change `imagetype='recrop'` in this line](https://github.com/datduong/ClassifyNF1/blob/master/Experiment/MakeScript.py#L28)

    - The video below show how you create the training script. 

    [![asciicast](https://asciinema.org/a/AFY2FSOtqXzWy0KSveqrP4UR2.svg)](https://asciinema.org/a/AFY2FSOtqXzWy0KSveqrP4UR2)

  - `Experiment/MakeScript.py` will also run on test images, and create the attribution plot for these test images, so you do not need to manually run testing and attribution. 
    - Attribution requires a lot of memory space, so make sure your machine has at least 64 GB RAM. 
    - To take an average attribution of a test image over the 5 folds, run `SeeAttribution/AverageAttrImg.py`, please accordingly change the folder [path here](https://github.com/datduong/ClassifyNF1/blob/master/SeeAttribution/AverageAttrImg.py#L15) and [here](https://github.com/datduong/ClassifyNF1/blob/master/SeeAttribution/AverageAttrImg.py#L19) to your folder on your own machine. 


### Hyperparameters

`Experiment/MakeScript.py` has for-loop to probe for several hyperparameters. We found that 
  - _focused images_ (or `Crop`) works best with `weight=5`, `schedulerscaler=10`, `learn_rate=3e-05`, and `dropout=0.2`. 
  - _panoramic images_ (or `Recrop`) works best with `weight=10`, `schedulerscaler=10`, `learn_rate=1e-05`, and `dropout=0.2`. 

## Ensemble classifier

We run a 5-fold cross-validation, and then ensemble these 5 models. You can use `ensemble.sh` and will not need to edit `ensemble_our_classifier.py`. 
  - Again, please change the [path in `ensemble.sh`](https://github.com/datduong/ClassifyNF1/blob/be84599a6ebde077a9a7f31b83c94bfc5fe4d631/ensemble.sh#L9) to your folder on your own machine. 

## Pre-trained models

[Pytorch pre-trained models (ends with .pth) are on this google drive here](https://drive.google.com/drive/folders/1m2c7uWPOkIK_FU3gTIpjJbqfpYHqE_0_?usp=sharing). In this example, I am put all the models into the folder `/data/duongdb/DeployOnline/ClassifyNF1/Example/`, but you can choose your own path on your own machine. 

After you download the pre-trained models, you can run `Experiment/MakeScriptEvalOnly.py` to make the evaluation scripts. *Please note, you have to change the fold paths with respect to on your own machine.* The video below shows you where to put the pre-trained models. We do 5 fold-cv so there will be 5 models. You can evaluate each of them like in the video below. 

[![asciicast](https://asciinema.org/a/3GGbptAuPaQ7HohAuLFq1wNaN.svg)](https://asciinema.org/a/3GGbptAuPaQ7HohAuLFq1wNaN)

