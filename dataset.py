import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

from tqdm import tqdm


class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None, transform_resize=None ):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform
        self.transform_resize = transform_resize

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]
        try: 
            image = cv2.imread(row.filepath) # ! read a file path from @csv, so don't need different folders, but have to reindex labels
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except: 
            print ('\npath not found\n' + row.filepath) 
            exit()

        image_resize = 0 # can't return none in class Data
        if self.transform_resize is not None: 
            res = self.transform_resize(image=image) # ! just resize, nothing more, need this to plot later
            image_resize = res['image'].astype(np.uint8) # https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

        # ! transform image for model
        res = self.transform(image=image) # ! uses albumentations
        image = res['image'].astype(np.float32)
        image = image.transpose(2, 0, 1) # makes channel x h x w instead of h x w x c

        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data, row.filepath, image_resize # ! return path so we can debug and plot attribution model ?
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long(), row.filepath, image_resize


def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    transforms_resize = albumentations.Compose([ # ! meant to used with attribution
        albumentations.Resize(image_size, image_size)
    ])

    return transforms_train, transforms_val, transforms_resize


def strong_aug(image_size): # https://github.com/albumentations-team/albumentations_examples
    return  albumentations.Compose([
            albumentations.RandomRotate90(),
            albumentations.Flip(),
            albumentations.Transpose(),
            albumentations.OneOf([
                albumentations.IAAAdditiveGaussianNoise(),
                albumentations.GaussNoise(),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.MotionBlur(p=0.2),
                albumentations.MedianBlur(blur_limit=3, p=0.1),
                albumentations.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            albumentations.OneOf([
                albumentations.OpticalDistortion(p=0.3),
                albumentations.GridDistortion(p=0.1),
                albumentations.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
                albumentations.RandomBrightnessContrast(),
            ], p=0.3),
            albumentations.HueSaturationValue(p=0.3),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize()
        ])

    
def get_transforms_celeb(image_size):

    transforms_train = strong_aug (image_size) # https://github.com/albumentations-team/albumentations_examples

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    transforms_resize = albumentations.Compose([ # ! meant to used with attribution
        albumentations.Resize(image_size, image_size)
    ])

    return transforms_train, transforms_val, transforms_resize


def get_meta_data(df_train, df_test):

    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)

    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features


# {'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'SCC': 10, 'VASC': 12, 'melanoma': 13, 'nevus': 14, 'unknown': 15}

def get_df(kernel_type, out_dim, data_dir, data_folder, use_meta, our_csv=None, img_map_file='train.csv', our_data_dir=None, celeb_data=None, coco_data=None):

    # 2020 data
    df_train = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'train.csv')) # ! do not change 'train.csv'
    df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True) # ! remove duplicated based on https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/train', f'{x}.jpg'))

    # ! @kernel_type contains the name "new_fold", @kernel_type is just their own naming convention
    if 'newfold' in kernel_type:
        tfrecord2fold = { # based on https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526
            8:0, 5:0, 11:0, # each record number is divided via "3-layer stratification"
            7:1, 0:1, 6:1,
            10:2, 12:2, 13:2,
            9:3, 1:3, 3:3,
            14:4, 2:4, 4:4,
        }
    elif 'oldfold' in kernel_type:
        tfrecord2fold = {i: i % 5 for i in range(15)}
    else:
        tfrecord2fold = {
            2:0, 4:0, 5:0,
            1:1, 10:1, 13:1,
            0:2, 9:2, 12:2,
            3:3, 8:3, 11:3,
            6:4, 7:4, 14:4,
        }
    df_train['fold'] = df_train['tfrecord'].map(tfrecord2fold)
    df_train['is_ext'] = 0

    # 2018, 2019 data (external data)
    df_train2 = pd.read_csv(os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}', 'train.csv')) # ! do not change 'train.csv'
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    # ! add path to actual images ... so we don't need to have different folders for each label class
    df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}/train', f'{x}.jpg'))
    if 'newfold' in kernel_type:
        df_train2['tfrecord'] = df_train2['tfrecord'] % 15 # ! there are 15 "buckets" in 2020, so they randomly assign ?
        df_train2['fold'] = df_train2['tfrecord'].map(tfrecord2fold)
    else:
        df_train2['fold'] = df_train2['tfrecord'] % 5
    df_train2['is_ext'] = 1 # ! external data (2019 and older)

    # Preprocess Target
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

    if out_dim >= 9:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    elif out_dim == 4:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('DF', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('AK', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('SCC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('VASC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('BCC', 'unknown'))
    else:
        raise NotImplementedError()

    # ! need to add our own data
    # create a text file, with diagnosis. 
    our_label_index = None
    our_label_names = None
    if our_csv is not None: 
        print ('add our own data')
        df_train, our_label_names = prepare_img_list (our_csv, kernel_type, tfrecord2fold, out_dim, 2, df_train, img_map_file=None, our_data_dir=our_data_dir) # our dataset is the 2nd dataset appended to 2020 data

    # ! add celeb data
    celeb_label_index = None
    if celeb_data is not None: 
        print ('add celeb data')
        df_train, celeb_label_names = prepare_img_list (celeb_data, kernel_type, tfrecord2fold, out_dim, 3, df_train, img_map_file=None) 

    # ! add coco person data
    coco_label_index = None
    if coco_data is not None: 
        print ('add coco data')
        df_train, coco_label_names = prepare_img_list (coco_data, kernel_type, tfrecord2fold, out_dim, 4, df_train, img_map_file=None) 

    # concat train data
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

    # ! count each disease in train 
    print ( 'df train labels {}'.format ( df_train.groupby('diagnosis').count() ) ) 
    
    # test data
    df_test = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/test', f'{x}.jpg'))

    if use_meta:
        df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)
    else:
        meta_features = None
        n_meta_features = 0

    # class mapping
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
    print ('diagnosis2idx {}'.format(diagnosis2idx))
    df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
    # mel_idx = diagnosis2idx['melanoma']

    # ! return index of our labels
    if our_csv is not None: 
        our_label_index = [ diagnosis2idx[s] for s in our_label_names ]
    # ! return celeb index
    if celeb_data is not None: 
        celeb_label_index = [ diagnosis2idx[s] for s in celeb_label_names ]
    # ! return coco index
    if coco_data is not None: 
        coco_label_index = [ diagnosis2idx[s] for s in coco_label_names ]

    return df_train, df_test, meta_features, n_meta_features, our_label_names, our_label_index, diagnosis2idx, celeb_label_index, coco_label_index


def prepare_img_list (data_dir, kernel_type, tfrecord2fold, out_dim, is_ext_code, df_train, img_map_file, our_data_dir=None): 

    if img_map_file is None: 
        df_train2 = pd.read_csv(data_dir)
    else: 
        df_train2 = pd.read_csv(os.path.join(data_dir, img_map_file))
        
    # df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True) # only used to remove stuffs tfrecord<0 was duplicated images
    # ! add path to actual images ... so we don't need to have different folders for each label class
    print ('our data columns')
    data_colnames = df_train2.columns.tolist()
    print (data_colnames)
    
    # if 'filepath' not in data_colnames: # we can pre-specify path name
    #     print ('make a filepath for our data, legacy??')
    #     df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir, 'train', f'{x}.jpg'))
    
    if 'newfold' in kernel_type: # ! using the newest fold, split based on how many img a person has. 
        df_train2['tfrecord'] = df_train2['tfrecord'] % 15 # ! there are 15 "buckets" in 2020, so they randomly assign ?
        df_train2['fold'] = df_train2['tfrecord'].map(tfrecord2fold)
    else:
        df_train2['fold'] = df_train2['tfrecord'] % 5

    df_train2['is_ext'] = is_ext_code # ! use this later to analyze results

    # ! add in path name, we se "filepath" as just the name, and not the full name
    if our_data_dir is not None: 
        df_train2['filepath'] = df_train2['filepath'].apply(lambda x: os.path.join(our_data_dir, f'{x}'))
    
    # ! fix the test set into a 6th fold. fold id=5
    if 'is_test' in data_colnames: 
        print ('put test set into 6th fold. fold id=5')
        df_train2.loc [ df_train2['is_test'] == 1 , 'fold' ] = 5 # https://stackoverflow.com/questions/36909977/update-row-values-where-certain-condition-is-met-in-pandas/36910033

    if 'target_celeb' in data_colnames: 
        target_multilabel = []
        for index, row in df_train2.iterrows(): 
            target_multilabel.append ( [float(n) for n in row['target_celeb'].split(';')] ) 
        # replace
        df_train2['target_celeb'] = target_multilabel

    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True) ## concat to larger data
    our_label_names = sorted ( list ( set ( df_train2['diagnosis'] ) ) ) ## call @df_train2
    return df_train, our_label_names


class SkinCelebData (Dataset): 
    def __init__(self, csv, mode, transform=None, transform_resize=None, args=None ):

        self.csv = csv.reset_index(drop=True) # takes in a csv.
        self.mode = mode
        self.transform = transform
        self.transform_resize = transform_resize
        self.celeb_num_label = len(args.celeb_label)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        # ! use the same idea as skin conditions
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath) 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_resize = 0 # can't return none in class Data
        if self.transform_resize is not None: 
            res = self.transform_resize(image=image) # ! just resize, nothing more, need this to plot later
            image_resize = res['image'].astype(np.uint8) # https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

        # ! transform image for model
        res = self.transform(image=image) # ! uses albumentations
        image = res['image'].astype(np.float32)
        image = image.transpose(2, 0, 1) # makes channel x h x w instead of h x w x c

        data = torch.tensor(image).float()

        # ! 
        label_skin = torch.tensor(self.csv.iloc[index].target).long() # single number, use 0 for celeb. # ! we will filter later
        # we should make the label in this form 0,1,1,0 ... so use split, and then convert to number
        if np.isnan(row.is_celeb): # we willl add a is_celeb in our csv file
            label_celeb = torch.zeros(self.celeb_num_label).float() # array
            is_celeb = 0
        else: 
            # temp = [float(j) for j in self.csv.iloc[index].target_celeb.split(',')]
            # self.csv.iloc[index].target_celeb should be already in np.array, we can do this ahead of time
            label_celeb = torch.tensor(self.csv.iloc[index].target_celeb).float() # takes [0,1,1,0] format for multilabel
            is_celeb = 1
            
        if self.mode == 'test':
            return data, row.filepath, image_resize # ! return path so we can debug and plot attribution model ?
        else:
            return data, label_skin, row.filepath, image_resize, label_celeb, is_celeb
        

class CelebData (Dataset): 
    def __init__(self, csv, mode, transform=None, transform_resize=None, args=None ):

        self.csv = csv.reset_index(drop=True) # takes in a csv.
        self.mode = mode
        self.transform = transform
        self.transform_resize = transform_resize
        self.celeb_num_label = len(args.celeb_label)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        # ! use the same idea as skin conditions
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath) 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_resize = 0 # can't return none in class Data
        if self.transform_resize is not None: 
            res = self.transform_resize(image=image) # ! just resize, nothing more, need this to plot later
            image_resize = res['image'].astype(np.uint8) # https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

        # ! transform image for model
        res = self.transform(image=image) # ! uses albumentations
        image = res['image'].astype(np.float32)
        image = image.transpose(2, 0, 1) # makes channel x h x w instead of h x w x c
        data = torch.tensor(image).float()

        # ! 
        label_celeb = torch.tensor(self.csv.iloc[index].target_celeb).float() # takes [0,1,1,0] format for multilabel
        # is_celeb = 1
            
        if self.mode == 'test':
            return data, row.filepath, image_resize # ! return path so we can debug and plot attribution model ?
        else:
            return data, label_celeb, row.filepath, image_resize
        
