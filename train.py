import os, json
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
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
import apex
from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma

import OtherMetrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    # !
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--n-test', type=int, default=1, help='how many times do we flip images, 1=>no_flip, max=8')
    parser.add_argument('--scheduler-scaler', type=float, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no-scheduler', action='store_true', default=False)
    parser.add_argument('--our-csv', type=str, default=None)
    parser.add_argument('--our-data-dir', type=str, default=None)
    parser.add_argument('--celeb-data', type=str, default=None)
    parser.add_argument('--coco-data', type=str, default=None)
    parser.add_argument('--weighted-loss', type=float, default=1)
    parser.add_argument('--weighted-loss-ext', type=float, default=1) ## decrease external weights
    parser.add_argument('--img-map-file', type=str, default='train.csv')
    parser.add_argument('--loaded-model', type=str, default=None)
    
    # args, new_args = parser.parse_known_args()
    args = parser.parse_args() # ! safer 
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target, path, _) in bar: # added path name and original resize image (only needed during attribution)

        optimizer.zero_grad()
        
        if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data)        
        
        loss = criterion(logits, target)

        if not args.use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if args.image_size in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0: # ! return original if I = 0
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, our_label_index, is_ext=None, n_test=1, get_output=False, criterion=None):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target, path, _) in tqdm(loader): # added path name, and original resize
            
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

            # average over all the augmentation of test data
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100. # ! global accuracy
        bal_acc = OtherMetrics.compute_balanced_accuracy_score(PROBS, TARGETS)
        bal_acc_20 = OtherMetrics.compute_balanced_accuracy_score(PROBS[is_ext == 0], TARGETS[is_ext == 0])
        bal_acc_ourdata = 0
        if args.our_csv is not None: 
            bal_acc_ourdata = OtherMetrics.compute_balanced_accuracy_score(PROBS[is_ext == 2], TARGETS[is_ext == 2])  
        return val_loss, acc, bal_acc, bal_acc_20, bal_acc_ourdata


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, our_label_index, criterion):

    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold] # ! take out a fold and keep it as valid
        df_valid = df[df['fold'] == fold]
        print ('df input size {}'.format(df.shape[0]))
        print ('df_train input size after remove fold {} {}'.format(fold,df_train.shape[0]))
        print ('df_valid input size after remove fold {} {}'.format(fold,df_valid.shape[0]))

    # take out fold id=5, this is our testset id
    print ('len before remove fold id=5 {}'.format(df_train.shape[0]))
    df_train = df_train[df_train['fold'] != 5]
    print ('len after remove fold id=5 {}'.format(df_train.shape[0]))

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers//2)

    print('train and dev data size {} , {}'.format(len(dataset_train), len(dataset_valid)))

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim,
        pretrained=True, 
        args=args
    )

    model = model.to(device)

    # ! loading in a model
    if args.loaded_model is not None: 
        print ('\nloading {}\n'.format(args.loaded_model))
        try:  # single GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                # ! ! setting strict=False
                model.load_state_dict(torch.load(args.loaded_model), strict=True, map_location=torch.device('cpu')) # ! avoid error in loading model trained on GPU
            else: 
                model.load_state_dict(torch.load(args.loaded_model), strict=True) 
        except:  # multi GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(args.loaded_model, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(args.loaded_model)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            # "myfc_celeb.weight", "myfc_celeb.bias"
            if 'myfc_celeb.weight' in state_dict: # ! single train on skin images.
                del state_dict['myfc_celeb.weight']
                del state_dict['myfc_celeb.bias']
            model.load_state_dict(state_dict, strict=True)

    # ! send to multiple gpus... only works well if we have model.forward, don't change forward func.
    if DP:
        model = apex.parallel.convert_syncbn_model(model)

    acc_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_all_fold{fold}.pth')
    model_file_final = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    # ! our own label
    bal_acc_ourdata_max = 0. 
    model_file_our = os.path.join(args.model_dir, f'{args.kernel_type}_ourlabel_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)

    if not args.no_scheduler: 
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=args.scheduler_scaler, total_epoch=1, after_scheduler=scheduler_cosine)

    best_epoch = 0 # ! early stop 
    
    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
		# scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion=criterion)
        val_loss, acc, bal_acc, bal_acc_20, bal_acc_ourdata = val_epoch(model, valid_loader, our_label_index, is_ext=df_valid['is_ext'].values, n_test=args.n_test, criterion=criterion) # @df_valid['is_ext'].values array, so we can index @PROBS

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, bal_acc {(bal_acc):.6f}, bal_acc_20 {(bal_acc_20):.6f}, bal_acc_ourdata {(bal_acc_ourdata):.6f}'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        if not args.no_scheduler: 
            scheduler_warmup.step()    
            if epoch==2: scheduler_warmup.step() # bug workaround   
                
        if acc > acc_max: # ! save best model on all labels, they save 2 types of model, one on old data, the other year 2020 
            print('acc_max ({:.6f} --> {:.6f}). Saving model ...'.format(acc_max, acc))
            torch.save(model.state_dict(), model_file) ## @model is the same model in both cases, just eval them separately
            acc_max = acc
        
        # ! save model best for our data
        if bal_acc_ourdata > bal_acc_ourdata_max:
            print('bal_acc_ourdata_max ({:.6f} --> {:.6f}). Saving model ...'.format(bal_acc_ourdata_max, bal_acc_ourdata))
            torch.save(model.state_dict(), model_file_our)
            bal_acc_ourdata_max = bal_acc_ourdata
            best_epoch = epoch 

        # ! early stop based on acc. for our data
        if epoch - best_epoch > 10 : 
            break 
        
    # ! end loop
    torch.save(model.state_dict(), model_file_final)


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

    if args.weighted_loss > 1 : # ! bring @criterion so we can use @our_label_index
        weight = torch.ones(args.out_dim) 
        print ('our_label_index {}'.format(our_label_index)) # ! not smart but whatever, let's try for now
        weight [ our_label_index ] = weight [ our_label_index ] * args.weighted_loss # ! upweigh our labels contribution to loss function
        if celeb_label_index is not None: # ! how about celeb data ? 
            weight [ celeb_label_index ] = weight [ celeb_label_index ] * args.weighted_loss_ext # half contribution
        if coco_label_index is not None: # ! also download coco little bit
            weight [ coco_label_index ] = weight [ coco_label_index ] * args.weighted_loss_ext # half contribution
        # 
        criterion = nn.CrossEntropyLoss(weight=weight.to(device))
    else: 
        criterion = nn.CrossEntropyLoss()

    # !
    transforms_train, transforms_val, _ = get_transforms(args.image_size) # don't need 3rd transform resize 

    folds = [int(i) for i in args.fold.split(',')]
    print ('\nfolds {}\n'.format(folds))
    for fold in folds: # ! run many folds
        run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, our_label_index, criterion)
        

if __name__ == '__main__':

    args = parse_args()
    with open(os.path.join(args.log_dir,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed(seed=args.seed) # ! set a seed, default to 0

    device = torch.device('cuda')
   
    main()
