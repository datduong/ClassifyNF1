
import os, numpy, PIL 
import re
import numpy as np 
from PIL import Image


# ! because we use 5-fold cv. we can average attribution for each fold. 
# https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil

# Access all PNG files in directory
# allfiles=os.listdir(os.getcwd())
# imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG",".jpeg",".jpg"]]

maindir = '/data/duongdb/DeployOnline/ClassifyNF1/Example/b4ns448recropWl10ss10lr1e-05dp0.2b64ntest1/EvalDev'

for level in ['_test_1','_test_5','_test_10']:

  outdir = '/data/duongdb/DeployOnline/ClassifyNF1/Example/b4ns448recropWl10ss10lr1e-05dp0.2b64ntest1/AverageAttr'+level
  if not os.path.exists(outdir): 
    os.mkdir(outdir)

  # 
  fold = [ str(i) + level for i in np.arange(5)]

  os.chdir(maindir)

  imlist_in_1_fold = sorted ( os.listdir(os.path.join(maindir,fold[0])) ) 
  imlist_in_1_fold = [i for i in imlist_in_1_fold if 'Positive' in i] # Positive Sign

  # ! 
  # this_img = imlist_in_1_fold[0]
  for this_img in imlist_in_1_fold: 

    imlist = [os.path.join(maindir,i,this_img) for i in fold]

    # Assuming all images are the same size, get dimensions of first image
    w,h=Image.open(imlist[0]).size
    N=len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr=numpy.zeros((h,w,3),numpy.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
      imarr=numpy.array(Image.open(im),dtype=numpy.float)
      arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    out.save(os.path.join(outdir,re.sub(r"(\.png|\.jpg)","",this_img)) + "Average.png")
    # out.show()

