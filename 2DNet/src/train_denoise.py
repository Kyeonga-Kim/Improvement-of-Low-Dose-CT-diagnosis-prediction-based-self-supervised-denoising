import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data
import sys
import torch.utils.data as data
from torchvision.transforms.transforms import ToTensor
#from torchsummaryX.torchsummaryX import summary
from net.models import *

from dataset.dataset2 import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW, RAdam
from tuils.loss_function import *
import torch.nn.functional as F
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(1992)
torch.cuda.manual_seed(1992)
np.random.seed(1992)
random.seed(1992)
from PIL import ImageFile
import sklearn
import copy
torch.backends.cudnn.benchmark = True
import argparse
import torchvision.models as models

####Denoising######
from tuils.mask_deno import Masker

from noise2self.models.dncnn import DnCNN
#from noise2self.models.unet import Unet
from torch.nn import MSELoss
import torchvision.transforms as transforms
from skimage.measure import compare_psnr
from skimage import data, img_as_float, img_as_ubyte
#from tuils.util import compute_psnr
import torch.optim as optim

from tensorboardX import SummaryWriter
#from torchsummaryX import summary
summary = SummaryWriter('runs/otitis_denosing_batch16_layer12_512') #이전 : otitis_denosing_batch16_layer12_t

def epochVal(epochID, model_de, loss_de, val_loader, c_val, val_batch_size):
    model_de.eval()
    lossValNorm = 0
    val_losses = 0

    for i, input in enumerate(val_loader):
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')

        input = input.cuda()  

        masker = Masker(width=4, mode='interpolate')    
        net_input, mask = masker.mask(input, i)
        net_output = model_de(net_input)
        val_loss = loss_de(net_output*mask, input*mask)
        val_losses = val_losses + val_loss.item()

        #denoiser                     
        output_de = model_de(input)
        output_de = torch.clamp(output_de, 0.0, 1.0)

        lossValNorm += 1

    val_losses = val_losses / lossValNorm

    #accuracyscore(outGT, outPRED, c_val)

    #write in tensorboard 
    input_grid = torchvision.utils.make_grid(input) #마지막 배치 이미지만 시각화
    summary.add_image('original_images', input_grid, epochID)

    denoised_grid = torchvision.utils.make_grid(output_de)
    summary.add_image('denoised_images', denoised_grid, epochID)

    return val_losses

def train(image_size):
    # cuda visble devices 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']

    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    df_all = pd.read_csv(csv_path) #label 있는 파일

    f_train = open('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/filename_train_de.txt', 'r')   
    f_val = open('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/filename_val_de.txt', 'r') 
    c_train = f_train.readlines()
    c_val = f_val.readlines() 
    f_train.close()
    f_val.close()
    c_train = [s.replace('\n', '') for s in c_train]
    c_val = [s.replace('\n', '') for s in c_val]   

    #for debug
    # c_train = c_train[0:1000]
    # c_val = c_val[0:100]

    print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))

    train_transform, val_transform = generate_transforms(image_size)
    train_loader, val_loader = generate_dataset_loader2(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

    model_de = DnCNN(1, num_of_layers=12).cuda()
    # model_de = Unet().cuda()
    model_de = torch.nn.DataParallel(model_de)
    # optimizer = torch.optim.SGD(model_de.parameters(), momentum=0.9, lr=1e-2, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model_de.parameters(), lr=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_de = MSELoss().cuda()
    
    trMaxEpoch = 100    
    for epochID in range (trMaxEpoch):  
        start_time = time.time()
        model_de.train()

        scheduler.step()  

        train_losses = 0
        best_val_loss = 1

        lossTrainNorm = 0

        for batchID, input in enumerate(train_loader): #input : noisy image
            if batchID == 0:
                ss_time = time.time()

            print(str(batchID) + '/' + str(int(len(c_train)/train_batch_size)) + '     ' + str((time.time()-ss_time)/(batchID+1)), end='\r')

            input = input.cuda() # (16, 3, 512, 512)

            # t_input_grid = torchvision.utils.make_grid(input) #마지막 배치 이미지만 시각화
            # summary.add_image('train_aug_images', t_input_grid, batchID)

            # denoiser     
            masker = Masker(width=4, mode='interpolate')     
            net_input, mask = masker.mask(input, batchID) 

            output_de = model_de(net_input)
            lossvalue2 = loss_de(output_de*mask, input*mask) 
            train_losses = train_losses + lossvalue2.item()

            lossTrainNorm += 1
  
            lossvalue2.backward()
            optimizer.step()
            optimizer.zero_grad()  
    
        train_losses = train_losses / lossTrainNorm  

        val_losses = epochVal(epochID, model_de, loss_de, val_loader, c_val, val_batch_size)
        summary.add_scalar('denoised_loss/val', val_losses, epochID) # val loss : 10 epoch 마다               
        summary.add_scalar('denoised_loss/train', train_losses, epochID)

        result = [epochID,
            round(optimizer.state_dict()['param_groups'][0]['lr'], 6),
            'train_loss:', train_losses,
            'val_loss:', val_losses, 
            ]

        print(result)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)    
        
        # if (epochID+1) % 10 == 0:
        torch.save({'epoch': epochID + 1, 'state_dict': model_de.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, snapshot_path + '/model_epoch_' + str(epochID) + '.pth')     
       
    del model_de


if __name__ == '__main__':
    csv_path = '/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/otitis_cls_final_md.csv' #라벨에 맞춘 csv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='se_resnext101_32x4d_256', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=512, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=16, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=8, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='se_resnext101_32x4d/denoising_3bone_otitis_all_512_16_8_aug', help='epoch')
    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 24  #24
    backbone = args.backbone
    print('image size:', Image_size)
    print('train batch size:', train_batch_size)
    print('val batch size:', val_batch_size)
    snapshot_path = 'data_test/' + args.model_save_path.replace('\n', '').replace('\r', '')
    train(Image_size)
    # valid_snapshot(backbone, Image_size)
