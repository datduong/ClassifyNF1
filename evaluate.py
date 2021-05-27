import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2

import apex # ! can be hard to install
from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma
from train import get_trans

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel

from copy import deepcopy 

import OtherMetrics
from SeeAttribution import GetAttribution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final','best_all','ourlabel'], default="ourlabel") # "best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default=None) # '0'
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')
    # !
    parser.add_argument('--dropout', type=float, default=0.5) # doesn't get used
    parser.add_argument('--n-test', type=int, default=1, help='how many times do we flip images, 1=>no_flip, max=8')
    parser.add_argument('--our-csv', type=str, default=None)
    parser.add_argument('--our-data-dir', type=str, default=None)
    parser.add_argument('--celeb-data', type=str, default=None)
    parser.add_argument('--coco-data', type=str, default=None)
    parser.add_argument('--attribution_keyword', type=str, default=None) 
    parser.add_argument('--outlier_perc', type=int, default=10, help='show fraction of high contributing pixel, default 10%')
    parser.add_argument('--img-map-file', type=str, default='train.csv')
    parser.add_argument('--do_test', action='store_true', default=False)

    args = parser.parse_args()
    return args



def val_epoch(model, loader, our_label_index, diagnosis2idx, is_ext=None, n_test=1, get_output=True, fold=None, args=None):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []

    if args.attribution_keyword is not None: # ! if do attribution
        print ('\nwill do attribution_model\n')
        attribution_model = IntegratedGradients(model) # send back to cpu, cuda takes up too much space
        n_test = 1 # ! test original image, not flipping

    with torch.no_grad():
        for (data, target, path, data_resize) in tqdm(loader):

            if args.attribution_keyword is not None: 
                our_label_index = [path.index(j) for j in path if args.attribution_keyword in j] # get only NF1 or HMI or etc...
                if len (our_label_index) == 0 :
                    continue # skip
                    
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu()) # ! @PROBS is shape=(1, 11708, 11)
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

            # ! do attribution here. call IntegratedGradient, or some other approaches
            if args.attribution_keyword is not None: 
                our_label_index = [path.index(j) for j in path if args.attribution_keyword in j] # get only NF1
                if len (our_label_index) > 0 :
                    temp = GetAttribution.GetAttributionPlot (  data[our_label_index].detach().cpu(), 
                                                                probs[our_label_index].detach().cpu(), 
                                                                np.array(path)[our_label_index], 
                                                                data_resize[our_label_index], 
                                                                attribution_model, 
                                                                fold=fold,
                                                                true_label_index=target[our_label_index].detach().cpu(),
                                                                args=args)

    # ! end eval loop
    if args.attribution_keyword is not None: 
        exit() # ! just do attribution
        
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy() # ! @PROBS is shape=(1, 11708, 11)
    TARGETS = torch.cat(TARGETS).numpy()

    # ! compute acc for this fold. 
    
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    for key,idx in diagnosis2idx.items(): # ! AUC
        if idx in our_label_index: 
            auc = roc_auc_score((TARGETS == idx).astype(float), PROBS[:, idx]) # ! global
            auc_20 = roc_auc_score((TARGETS[is_ext == 2] == idx).astype(float), PROBS[is_ext == 2, idx]) # ! condition on our data (fewer samples)
            print (time.ctime() + ' ' + f'Fold {fold}, {key} auc: {auc:.5f}, acc_local: {auc_20:.4f}')

    # ! weighted accuracy
    bal_acc = OtherMetrics.compute_balanced_accuracy_score(PROBS, TARGETS)
    bal_acc_20 = OtherMetrics.compute_balanced_accuracy_score(PROBS[is_ext == 0], TARGETS[is_ext == 0])
    
    bal_acc_ourdata = 0
    if args.our_csv is not None: 
        print ('\n\nour data size {}\n\n'.format( PROBS[is_ext == 2].shape ))
        bal_acc_ourdata = OtherMetrics.compute_balanced_accuracy_score(PROBS[is_ext == 2], TARGETS[is_ext == 2])

    # ! global confusion matrix
    OtherMetrics.plot_confusion_matrix( PROBS, TARGETS, diagnosis2idx, os.path.join(args.log_dir,'confusion_matrix_fold'+str(fold) ), our_label_index )
    
    return LOGITS, PROBS, val_loss, acc, bal_acc, bal_acc_20, bal_acc_ourdata



def main():

    df, df_test, meta_features, n_meta_features, our_label_names, our_label_index, diagnosis2idx, celeb_label_index, coco_label_index = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta, 
        our_csv=args.our_csv,
        our_data_dir=args.our_data_dir,
        img_map_file=args.img_map_file,
        celeb_data=args.celeb_data,
        coco_data=args.coco_data
    )

    transforms_train, transforms_val, transforms_resize = get_transforms(args.image_size)

    ## see our input 
    print ('our_label_index {}'.format(our_label_index))
    temp = df[df.is_ext == 2].copy().reset_index(drop=True)
    print (temp)
    
    LOGITS = []
    PROBS = []
    for fold in [int(i) for i in args.fold.split(',')]:

        if args.do_test: 
            print ('\ntesting on fold id=5 using data trained without fold {}\n'.format(fold))
            df_valid = df[df['fold'] == 5] # ! eval on our own test set
            df_valid_with_ham10k = df[ (df['fold'] == fold) & (df['is_ext'] != 2) ] # is_ext=0 for 2020 competition, is_ext=1 for older data, is_ext=2 is our data
            df_valid = pd.concat( [df_valid,df_valid_with_ham10k] ) # we have to do this to "trick" the input which requires all the labels
        else: 
            df_valid = df[df['fold'] == fold] # ! eval on the left-out fold

        print ('eval data size {}'.format(df_valid.shape))

        dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val, transform_resize=transforms_resize)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        print ('len of valid pytorch dataset {}'.format(len(dataset_valid)))
        
        # ! load model         
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_{args.eval}_fold{fold}.pth')
        print ('\nmodel_file {}\n'.format(model_file))
        
        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim,
            pretrained=True, 
            args=args
        )
        model = model.to(device)

        try:  # single GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                model.load_state_dict(torch.load(model_file), strict=True, map_location=torch.device('cpu')) # ! avoid error in loading model trained on GPU
            else: 
                model.load_state_dict(torch.load(model_file), strict=True) 
        except:  # multi GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if DP : # len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        this_LOGITS, this_PROBS, val_loss, acc, bal_acc, bal_acc_20, bal_acc_ourdata = val_epoch(model, valid_loader, our_label_index, diagnosis2idx, is_ext=df_valid['is_ext'].values, n_test=args.n_test, get_output=True, fold=fold, args=args)
        LOGITS.append(this_LOGITS)
        PROBS.append(this_PROBS)

        # print 
        content = time.ctime() + ' ' + f'Fold {fold}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, bal_acc {(bal_acc):.6f}, bal_acc_20 {(bal_acc_20):.6f}, bal_acc_ourdata {(bal_acc_ourdata):.6f}'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_eval.txt'), 'a') as appender:
            appender.write(content + '\n')

        # ! merge data frame
        print ('PROB output size {}'.format(PROBS[0].shape))
        prob_df = pd.DataFrame( PROBS[0], columns=np.arange(args.out_dim) ) # @PROBS is shape=(1, 11708, 11)
        prob_df = prob_df.reset_index(drop=True) ## has to do this to concat right
        df_valid_temp = df_valid.copy()
        df_valid_temp = df_valid_temp.reset_index(drop=True)
        print ('dim df_valid_temp {} and prob_df {}'.format(df_valid_temp.shape,prob_df.shape))
        assert df_valid_temp.shape[0] == prob_df.shape[0]
        df_valid_prob = pd.concat([df_valid_temp, prob_df], axis=1) # ! just append col wise
        log_file_name = 'eval_fold_'+str(fold)+'.csv'
        if args.do_test:
            log_file_name = 'test_on_fold_5_from_fold_'+str(fold)+'.csv'
        df_valid_prob.to_csv(os.path.join(args.log_dir, log_file_name),index=False)

    # end folds  



if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    if args.CUDA_VISIBLE_DEVICES is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
        DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
        device = torch.device('cuda')
    else: 
        DP = False  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    main()
