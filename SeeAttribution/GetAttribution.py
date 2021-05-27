import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
import re
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def GetAttributionPlot (transformed_img, output, img_path, original_resize, attribution_model, fold, attribution_name='_ig', true_label_index=None,args=None): 

  # @attribution_name "integrated gradient"
  # @transformed_img batch x 3 x h x w (because RGB), after data augmentation, not just resize

  if true_label_index is None: 
    prediction_score, pred_label_idx = torch.topk(output, 1) # take top prediction, # ! notice @prediction_score is not used
    pred_label_idx.squeeze_() # ! batch x 1 array
  else: 
    pred_label_idx = true_label_index

  temp_ = "_test_" if args.do_test else "_" 
  folder_path = os.path.join ( args.oof_dir, str(fold) + temp_ + str(args.outlier_perc)) # save images in its own fold
  if not os.path.exists (folder_path):
    os.mkdir(folder_path)

  # ! check if file exists, and then skip. note we will always use batch=0
  outpath = re.sub ( r'\.(jpg|png|jpeg)', attribution_name+args.attribution_keyword+'_bysidePositive.jpg', img_path[0]).split('/')
  outpath = os.path.join(folder_path, outpath[-1])
  if os.path.exists(outpath):
    return 0

  # ! @n_steps will determine how much memory we need, low--> bad approx to the intergral, high--> too much mem
  # ! baseline is black image. We may try different baselines.
  attributions_ig = attribution_model.attribute(transformed_img, target=pred_label_idx, n_steps=75)
  
  default_cmap = LinearSegmentedColormap.from_list('custom blue', # white-->black color gradient scale
                                                  [ (0, '#ffffff'),
                                                    (0.5, '#000000'),
                                                    (1, '#000000')], N=256)

  # we can only plot each image one at a time

  for b in range(transformed_img.shape[0]): 

    # ! overlay (only work with positive contribution)
    image = viz.visualize_image_attr( np.transpose(attributions_ig[b].squeeze().cpu().detach().numpy(), (1,2,0)), # ! (224, 224, 3), h x w x chanels
                                      original_resize[b].detach().numpy(), # @original_resize is tensor, even when we kept it as numpy, so we have to convert to np
                                      method='masked_image', # ! masked_image is gradient x input
                                      cmap=default_cmap,
                                      show_colorbar=False,
                                      sign='positive',
                                      outlier_perc=args.outlier_perc) # ! default is 10%

    outpath = re.sub ( r'\.(jpg|png|jpeg)', attribution_name+args.attribution_keyword+'_overlay.jpg', img_path[b]).split('/')
    image[0].savefig( os.path.join(folder_path, outpath[-1]) , bbox_inches='tight', pad_inches=0.0) 

    # ! side by side sign contribution
    image = viz.visualize_image_attr_multiple(np.transpose(attributions_ig[b].squeeze().cpu().detach().numpy(), (1,2,0)),
                                              original_resize[b].detach().numpy(),
                                              ["original_image", "heat_map"],
                                              ["all", "all"],
                                              # cmap=default_cmap,
                                              show_colorbar=True, 
                                              outlier_perc=args.outlier_perc)

    outpath = re.sub ( r'\.(jpg|png|jpeg)', attribution_name+args.attribution_keyword+'_bysideSign.jpg', img_path[b]).split('/')
    image[0].savefig( os.path.join(folder_path, outpath[-1]) , bbox_inches='tight', pad_inches=0.0) 

    # ! side by side positive only
    image = viz.visualize_image_attr_multiple(np.transpose(attributions_ig[b].squeeze().cpu().detach().numpy(), (1,2,0)),
                                              original_resize[b].detach().numpy(),
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              cmap=default_cmap,
                                              show_colorbar=True, 
                                              outlier_perc=args.outlier_perc)

    outpath = re.sub ( r'\.(jpg|png|jpeg)', attribution_name+args.attribution_keyword+'_bysidePositive.jpg', img_path[b]).split('/')
    image[0].savefig( os.path.join(folder_path, outpath[-1]), bbox_inches='tight', pad_inches=0.0) 

  # end. 
  return 1 
