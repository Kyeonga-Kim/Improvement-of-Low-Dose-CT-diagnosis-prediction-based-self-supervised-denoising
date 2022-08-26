import torch.utils.data as data
import torch
import albumentations
import cv2
import numpy as np
import random
import math
from settings import train_png_dir, img_path

#####################
import joblib
import PIL
from glob import glob
import pydicom
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import math
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import re
import logging as l
from glob import glob
import argparse
import torchvision.transforms as transforms
from skimage import img_as_float

def generate_transforms(image_size):
    IMAGENET_SIZE = image_size

    train_transform = albumentations.Compose([

        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=1, border_mode=4, always_apply=False, p=0.5),
        albumentations.RandomBrightness(limit=0.2, always_apply=False, p=0.5)

    ])

    val_transform = albumentations.Compose([
        # albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
        # albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0, p=1.0),
        # albumentations.Normalize(max_pixel_value=255.0, p=1.0)
    ])

    return train_transform, val_transform

# def generate_transforms(image_size):


#     train_transform = transforms.Compose([
#         # transforms.Normalize((0.456, 0.456, 0.456), (0.224, 0.224, 0.224))
 
#     ])

#     val_transform = transforms.Compose([
#         # transforms.Normalize((0.456, 0.456, 0.456), (0.224, 0.224, 0.224))
#     ])

#     return train_transform, val_transform



def generate_random_list(length):
    new_list = []

    for i in range(length):
        if i <= length/2:
            weight = int(i/4)
        else:
            weight = int((length - i)/4)
        weight = np.max([1, weight])
        new_list += [i]*weight

    return new_list    


# 3 window dicom -> png -> concat 

# dicom train 폴더 불러오기 -> dicom.SOPInstanceID이랑 같은 거  찾기. 
class RSNA_Dataset_train_window3_by_study_context(data.Dataset): 
    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list #fold별 저장해논 train filename
        self.transform = transform

   
    def __getitem__(self, idx):
        filename = self.name_list[idx % len(self.name_list)] # filename #00104723_10_20190925.png 

        # 환자번호 + 수술 날짜 조회된 df   
        filename_train_df = self.df[(self.df['unit_no']==int(filename.split('_')[-3])) & (self.df['inspection_date']==int(filename.split('_')[-1].rstrip('.png')))]          
        study_index = int(filename.split('_')[-2]) # slice num
        last_index = filename_train_df.iloc[:,8:].T.dropna(axis=0).shape[0]+10

        if study_index == last_index: #slice id가 끝번이면 
            filename_up = filename
   
        else:
            filename_up = filename.split('_')[-3] + '_' + str(study_index+1) + '_' + filename.split('_')[-1]
 
        if study_index == 11: #slice id가 시작 번호 이면
            filename_down = filename #slice id가 0이면 filename_down으로 시작 #00104723_10_20190925.png

        else:
            filename_down = filename.split('_')[-3] + '_' + str(study_index-1) + '_' + filename.split('_')[-1]

        # print('filename {}'.format(filename))
        # print('filename_up {}'.format(filename_up))
        # print('filename_down {}'.format(filename_down))
        image = cv2.imread('/home/kka0602/nas/otitis_bone/' + filename, 0)
        image = cv2.resize(image, (256, 256), cv2.INTER_AREA) #bone

        image_up = cv2.imread('/home/kka0602/nas/otitis_bone/' + filename_up, 0)
        image_up = cv2.resize(image_up, (256, 256), cv2.INTER_AREA)

        image_down = cv2.imread('/home/kka0602/nas/otitis_bone/' + filename_down, 0)
        image_down = cv2.resize(image_down, (256, 256), cv2.INTER_AREA)

        image_cat = np.concatenate([image_up[:,:,np.newaxis], image[:,:,np.newaxis], image_down[:,:,np.newaxis]], 2) #(512,512,3)

        #수정 @@@@
        label = torch.FloatTensor([filename_train_df[str(study_index)].values]) 
        

        image_cat, label = aug_image(image_cat, label, is_infer=False)
        #image_cat = cv2.resize(image_cat, (512, 512)) #bone
        image_cat = img_as_float(image_cat)     
        # image_cat = torch.Tensor(image_cat[np.newaxis])

        image_cat = torch.Tensor(image_cat.transpose(2, 0, 1)) # (3,512,512)

        # image_cat = PIL.Image.fromarray(image_cat)  
        # image_cat.save('/home/kka0602/nas/test/' + filename)
            
        # if self.transform is not None:
        #     augmented = self.transform(image=image_cat)     
        #     #image_cat = img_as_float(augmented['image'])
        #     image_cat = torch.FloatTensor(augmented['image'])
        #     #image_cat = image_cat.transpose(2, 0, 1) #(9,256,256)
        #     #print(image_cat.shape)
        #     image_cat = image_cat.transpose(2, 0, 1) #(9,256,256)


        return image_cat, label
    
    def __len__(self):
        return len(self.name_list)


class RSNA_Dataset_val_window3_by_study_context(data.Dataset):
    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list #fold별 저장해논 train filename
        self.transform = transform


    def __getitem__(self, idx):
        filename = self.name_list[idx % len(self.name_list)] # filename #00104723_10_20190925.png #24개  
        #filename_date = re.split(r'[.]', filename.split('_')[-1])
       # print(filename_date)

        # 환자번호 + 수술 날짜 조회된 df   
        filename_train_df = self.df[(self.df['unit_no']==int(filename.split('_')[-3])) & (self.df['inspection_date']==int(filename.split('_')[-1].rstrip('.png')))]       
        study_index = int(filename.split('_')[-2]) # slice num
        last_index = filename_train_df.iloc[:,8:].T.dropna(axis=0).shape[0]+10

        if study_index == last_index: #slice id가 끝번이면 
            filename_up = filename
   
        else:
            filename_up = filename.split('_')[-3] + '_' + str(study_index+1) + '_' + filename.split('_')[-1]
 
        if study_index == 11: #slice id가 시작 번호 이면
            filename_down = filename #slice id가 0이면 filename_down으로 시작 #00104723_10_20190925.png

        else:
            filename_down = filename.split('_')[-3] + '_' + str(study_index-1) + '_' + filename.split('_')[-1]

        # print('filename {}'.format(filename))
        # print('filename_up {}'.format(filename_up))
        # print('filename_down {}'.format(filename_down))
              
        image = cv2.imread('/home/kka0602/nas/otitis_bone/' + filename, 0)
        image = cv2.resize(image, (256, 256), cv2.INTER_AREA) #bone

        image_up = cv2.imread('/home/kka0602/nas/otitis_bone/' + filename_up, 0)
        image_up = cv2.resize(image_up, (256, 256), cv2.INTER_AREA)

        image_down = cv2.imread('/home/kka0602/nas/otitis_bone/' + filename_down, 0)
        image_down = cv2.resize(image_down, (256, 256), cv2.INTER_AREA)

        image_cat = np.concatenate([image_up[:,:,np.newaxis], image[:,:,np.newaxis], image_down[:,:,np.newaxis]], 2) #(512,512,3)

        #수정 @@@@
        label = torch.FloatTensor([filename_train_df[str(study_index)].values]) 

        image_cat, label = aug_image(image_cat, label, is_infer=True)
        # image_cat = cv2.resize(image_cat, (512, 512)) #bone
        image_cat = img_as_float(image_cat)
        # image_cat = torch.Tensor(image_cat[np.newaxis])
        image_cat = torch.Tensor(image_cat.transpose(2, 0, 1))


        return image_cat, label

    def __len__(self):
        return len(self.name_list)

def randomHorizontalFlip(image, label, u=0.5):
    if np.random.random() < u:
        label = np.asarray(label)
        image = cv2.flip(image, 1)
        label[0][0], label[0][1] = label[0][1], label[0][0]  # 0이면 1, 1이면 0
        label = torch.FloatTensor(label)
    return image, label
    

def randomVerticleFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)

    return image

def randomRotate90(image, u=0.5):
    if np.random.random() < u:
        image[:,:,0:3] = np.rot90(image[:,:,0:3])
    return image

#===================================================origin=============================================================
def random_cropping(image, ratio=0.8, is_random=True):
    height, width, _ = image.shape # 
    target_h = int(height*ratio) # 400
    target_w = int(width)

    if is_random:

        start_x = 0
        start_y = random.randint((height - target_h) // 4, (height - target_h) // 2) # 25 ~ 50

    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def cropping(image, ratio=0.8, code = 5):
    height, width, _ = image.shape
    target_h = int(height*ratio) #512*0.8 
    target_w = int(width) #512*0.8

    if code==0:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    elif code == 1:
        start_x = 0
        start_y = 0

    elif code == 2:
        start_x = width - target_w
        start_y = 0

    elif code == 3:
        start_x = 0
        start_y = height - target_h

    elif code == 4:
        start_x = width - target_w
        start_y = height - target_h

    elif code == 5 : 
        start_x = 0
        start_y = ( height - target_h ) // 2   

    elif code == -1:
        return image

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:] # (409,512,3)
    zeros = cv2.resize(zeros ,(width,height)) # (512, 512, 3)
    
    return zeros

def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, :] = 0.0
            else:
                print('!!!!!!!! random_erasing dim wrong!!!!!!!!!!!')
                return

            return img
    return img

def randomShiftScaleRotate(image,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):

    if np.random.random() < u:
        height, width, _ = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
    return image


def aug_image(image, label, is_infer=False):
    if is_infer:
        # image, label = randomHorizontalFlip(image, label, u=0) #좌우반전 안함.
        # image = np.asarray(image)      
        image = cropping(image, ratio=0.8, code=5)
        #image = random_cropping(image, ratio=0.8, is_random = True)

        return image, label

    else:
        image, label = randomHorizontalFlip(image, label, u=0.5)
        height, width, _ = image.shape
        image = randomShiftScaleRotate(image,
                                            shift_limit=(-0.1,  0.1),
                                            scale_limit=(-0.1, 0.1),
                                            aspect_limit=(-0.1, 0.1),
                                            rotate_limit=(-30, 30))

        image = cv2.resize(image, (width, height))
        ratio = random.uniform(0.6,0.9)
        image = random_cropping(image, ratio=ratio,  is_random=True)  

        #image = random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3)


        return image, label


def generate_dataset_loader3(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):
    train_dataset = RSNA_Dataset_train_window3_by_study_context(df_all, c_train, train_transform)
    val_dataset = RSNA_Dataset_val_window3_by_study_context(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size, #16       
        shuffle=True, # default : shuffle=True
        num_workers=workers,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size, #8  
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, val_loader
