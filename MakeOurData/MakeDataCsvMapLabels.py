
# ! we don't have meta data, so just put some fake number
# image_name,patient_id,sex,age_approx,anatom_site_general_challenge,diagnosis,benign_malignant,target,tfrecord,width,height
# ISIC_0000000,-1,female,55.0,anterior torso,NV,benign,0,4,1022,767
# ISIC_0000001,-1,female,30.0,anterior torso,NV,benign,0,18,1022,767
# ISIC_0000002,-1,female,60.0,upper extremity,MEL,malignant,1,0,1022,767

# ! images should be size 512 x 512 ?? not really, the train.csv has original size. 
# ! we didn't see major improvement when using image-super-resolution. 
# ! well ... we will convert to 512 x 512, interpolation = cv2.INTER_AREA

# pull out images from powerpoint, resize interpolation = cv2.INTER_AREA
# name images with disease labels, put labels on 1st page of power point 
# when create csv, can use label name 
# fill in random values for other fields
# image names are not important
# 5 folds, so we randomly assign the @tfrecord ? just do row count as @tfrecord then mod 5
# we assume we have pure randomness

import os,sys,re,pickle
import pandas as pd 
import numpy as np 

imagetype = '_recrop'
# imagetype = ''

if imagetype == '_recrop': 
  datapath = '/data/duongdb/DeployOnline/ClassifyNF1/Example/Recrop'
else: 
  datapath = '/data/duongdb/DeployOnline/ClassifyNF1/Example/Crop'


test_img_pickle = '/data/duongdb/DeployOnline/ClassifyNF1/Example/QualtricFolderImage/recrop_to_crop_map_testimg.pickle'

all_labels = 'IP HMI NF1 TSC MA EverythingElse ML'.split()

# @fout is a csv with the same format used to train skin cancer images in 2020
fout = open (datapath+"/train.csv","w")
fout.write("image_name,filepath,patient_id,sex,age_approx,anatom_site_general_challenge,diagnosis,benign_malignant,target,tfrecord,is_test,width,height\n")

all_img = []
for lab in all_labels: 
  img_this_lab = os.listdir( os.path.join( datapath,lab+imagetype,'TrimWhiteSpaceNoBorder') ) 
  all_img = all_img + [i for i in img_this_lab] # 'crop'+

#
print (len(all_img))

# ! 
np.random.seed(0)
all_img = np.random.permutation(all_img).tolist() ## ! need random swap add in some randomness when making folds
print (all_img[0:10])

# ! we now have a test set, so we don't do pure random split anymore 
test_img_dict = pickle.load(open(test_img_pickle,'rb')) # ! mapping is whole-->cut
test_img_to_use = []
for lab in all_labels: 
  if imagetype=='': # cut 
    arr = [ lab+'Slide'+str(v)+'.jpg' for k,v in test_img_dict[lab].items() ] # ! mapping is whole-->cut, name example: EverythingElse_recropSlide204.jpg
  else: 
    arr = [ lab+imagetype+'Slide'+str(k)+'.jpg' for k,v in test_img_dict[lab].items() ] # ! mapping is whole-->cut
  #
  test_img_to_use = test_img_to_use + arr 


print ('num test img {}'.format(len(test_img_to_use)))

# ! write a single csv containing all the names, so we don't have to copy/paste images
counter = 0
for index, img in enumerate ( all_img ) : 
  #
  if img in test_img_to_use: 
    this_index = '-2' # ! special fold for testing. train on fold 0-4
    is_test = '1'
  else: 
    this_index = counter
    is_test = '0'
    counter = counter + 1 # update counter only when see train images
  
  if "HMI" in img: 
    imgpath = os.path.join( datapath,'HMI'+imagetype,'TrimWhiteSpaceNoBorder',img)
    fout.write(img+','+imgpath+',-1,0,0,none,HMI,benign,0,'+str(this_index)+','+is_test+',512,512\n') ## all the other fields don't matter
  if "IP" in img: 
    imgpath = os.path.join( datapath,'IP'+imagetype,'TrimWhiteSpaceNoBorder',img)
    fout.write(img+','+imgpath+',-1,0,0,none,IP,benign,0,'+str(this_index)+','+is_test+',512,512\n')
  if "MA" in img: 
    imgpath = os.path.join( datapath,'MA'+imagetype,'TrimWhiteSpaceNoBorder',img)
    fout.write(img+','+imgpath+',-1,0,0,none,MA,benign,0,'+str(this_index)+','+is_test+',512,512\n')
  if "NF1" in img: 
    imgpath = os.path.join( datapath,'NF1'+imagetype,'TrimWhiteSpaceNoBorder',img)
    fout.write(img+','+imgpath+',-1,0,0,none,NF1,benign,0,'+str(this_index)+','+is_test+',512,512\n')
  if "TSC" in img: 
    imgpath = os.path.join( datapath,'TSC'+imagetype,'TrimWhiteSpaceNoBorder',img)
    fout.write(img+','+imgpath+',-1,0,0,none,TSC,benign,0,'+str(this_index)+','+is_test+',512,512\n')
  if "EverythingElse" in img: 
    imgpath = os.path.join( datapath,'EverythingElse'+imagetype,'TrimWhiteSpaceNoBorder',img)
    fout.write(img+','+imgpath+',-1,0,0,none,EverythingElse,benign,0,'+str(this_index)+','+is_test+',512,512\n')
  if "ML" in img: 
    imgpath = os.path.join( datapath,'ML'+imagetype,'TrimWhiteSpaceNoBorder',img)
    fout.write(img+','+imgpath+',-1,0,0,none,ML,benign,0,'+str(this_index)+','+is_test+',512,512\n')


# 
fout.close()
print ('total num img {}'.format(len(all_img)))

#
