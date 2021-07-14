import sys, os, re, pickle
import pandas as pd
import numpy as np
from glob import glob
import argparse

import OtherMetrics

def read_output(filename,args): 
    # we need to worry only about our iamges
    df = pd.read_csv(filename) # image_name,patient_id,sex,age_approx,anatom_site_general_challenge,diagnosis,benign_malignant,target,tfrecord,width,height,filepath,fold,is_ext,is_test,0,1,2,3,...
    df = df.sort_values(by='image_name',ignore_index=True) # sort just to be consisent. 
    p = r'({})'.format('|'.join(map(re.escape, args.labels))) # https://stackoverflow.com/questions/11350770/select-by-partial-string-from-a-pandas-dataframe
    df = df[df['image_name'].str.contains(p)]
    df = df.reset_index()
    print ('\nread in {} dim {}'.format(filename,df.shape[0]))
    print (df)
    return df


def rm_lt50_average(prediction_array,num_labels): # @prediction_array is [[model1], [model2]...] for one single observation 
    counter = 0
    ave_array = np.zeros(num_labels)
    for array in prediction_array: 
        if max(array) > 0.5 : # skip if no prediction is over 0.5
            ave_array = ave_array + array 
            counter = counter + 1
    #
    return ave_array/counter 
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--labels', type=str)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--NF1', type=int, default=0) # what label index to keep
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    args.labels = args.labels.strip().split(',')
    num_labels = len(args.labels)
    label_col_index = [str(i) for i in range(num_labels)] # ! col names are index, and not label names
    
    # outputs = [pd.read_csv(csv) for csv in sorted(glob(os.path.join(args.model_dir, '*csv')))] # read each prediction
    # sub_probs = [sub.target.rank(pct=True).values for sub in subs] # for each df, rank the target and convert to percent. 
    # @sub is obs x target. rank @sub by col "target". so we scale max prediction to 1, lowest to 0?
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html
    # wts = [1/18]*18
    # assert len(wts)==len(sub_probs)
    # sub_ens = np.sum([wts[i]*sub_probs[i] for i in range(len(wts))],axis=0)
    

    # ! we need to rank a prediction probability for each condition
    # ISIC2020 competition cares for just 1 condition. 
    outputs = [read_output(csv,args) for csv in sorted(glob(os.path.join(args.model_dir, 'test_on_fold_5_from_fold*csv')))] # read each prediction

    final_df = outputs[0] # place holder 
    prediction_np = np.zeros((final_df.shape[0],num_labels))
    
    for index in range(final_df.shape[0]) : 
        # average ?
        prediction_array = []
        for df1 in outputs: # go over each csv output, and get the row
            array1 = list ( df1.loc[index,label_col_index] ) 
            prediction_array.append ( [float(a) for a in array1] ) # array of array of numbers. [ [1,2,3], [2,3,4] ... ]
        # 
        prediction_np[index] = rm_lt50_average(prediction_array,num_labels) # ensemble over many models for 1 observation 
       

    print (prediction_np)
    print (prediction_np.shape)
    final_df.loc[:,label_col_index] = prediction_np
    
    # need to recode the truth and prob labels. 
    diagnosis2idx = {value:index for index,value in enumerate(args.labels)}
    if args.NF1 > 0: 
        our_label_index = np.array([4, 5, 6, 7, 8, 9, 11])
    else: 
        our_label_index = np.arange(len(args.labels))

    final_df['true_label_index'] = final_df['diagnosis'].map(diagnosis2idx)
    
    PROBS = prediction_np.argmax(axis=1)
    final_df['predict_label_index'] = PROBS
    
    print (PROBS)
    print (PROBS.shape)
    for p in PROBS: 
        if p not in our_label_index: 
            print ('we predict something outside our list of labels')
            print (p)
    
    TARGETS = np.array ( list (final_df['true_label_index']) ) 
    # print (TARGETS)
    
    OtherMetrics.plot_confusion_matrix_manual( prediction_np, TARGETS, diagnosis2idx, os.path.join(args.model_dir,'ensem_confusion_matrix'+ '' if args.NF1 else '_all' ), our_label_index )
    temp_ = '' if args.NF1>0 else '_all'
    final_df.to_csv(os.path.join(args.model_dir,"final_prediction"+temp_+".csv"),index=False) # writeout

    # ! compute average bal. acc. 
    print ('print (prediction_np.shape)', prediction_np.shape)
    balacc = OtherMetrics.compute_balanced_accuracy_score(prediction_np, TARGETS) # ! for our data only ?
    print ('bal. acc. ', balacc)

    acc = (prediction_np.argmax(1) == TARGETS).mean() * 100. # ! global accuracy
    print ('pure acc. ', balacc)
