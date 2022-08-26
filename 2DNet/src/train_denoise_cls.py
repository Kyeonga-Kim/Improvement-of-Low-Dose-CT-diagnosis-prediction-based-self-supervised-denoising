import os
import time
#from wsgiref.types import InputStream
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
import torch.utils.data
import sys
import torch.utils.data as data
from torchvision.transforms.transforms import ToTensor
#from torchsummaryX.torchsummaryX import summary
from net.models import *
from net.ConvNeXt import convnext_base

from dataset.dataset_cls import *
#from dataset.dataset import *
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

# from noise2self.models.unet import Unet
from noise2self.models.dncnn import DnCNN
from noise2self.models.red_cnn import RED_CNN

from torch.nn import MSELoss
import torchvision.transforms as transforms
#from skimage.measure import compare_psnr
from skimage import data, img_as_float, img_as_ubyte
#from tuils.util import compute_psnr


#from torchsummaryX import summary
from tensorboardX import SummaryWriter
summary = SummaryWriter('runs/otitis_denosing_cls_256_3concat_de_f') 

def space_to_depth(x, block_size):
    n, c, h, w = x.size() # 32, 3, 512, 512 -> 32 ,48, 128, 128
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

def add_pr_curve_tensorboard(test_probs, test_label, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    classes = ['right', 'left', 'any']
    for i in range(len(classes)):
        tensorboard_truth = test_label[:, i] # 0, 1, 2
        tensorboard_probs = test_probs[:, i]      

        summary.add_pr_curve(classes[i], #0~2
                            tensorboard_truth,
                            tensorboard_probs,
                            global_step=global_step)
        summary.close()
        
        
def epochVal(epochID, model, model_de, loss_cls, dataLoader, c_val, val_batch_size):

    model.eval()
    model_de.eval()

    lossValNorm = 0
    valLoss = 0

    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()

    for i, (input, target) in enumerate (dataLoader): # i : batch
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')

        #denoiser     
        input = input.cuda()     
             
        output_de_concat = []
        for i in range(3):
            image = input[:,i:i+1,:,:]                                            
            output_de = model_de(image)
            output_de = torch.clamp(output_de, 0.0, 1.0) # adjcent 3 window denoising 시킨후에 concat
            output_de_concat.append(output_de)

        output_de_concat = torch.cat(output_de_concat, dim=1)

        target = target.view(-1, 3).contiguous().cuda()
        #target = target[:,:3] #첫번째 class any만 가져오기.
        outGT = torch.cat((outGT, target), 0)

        varInput = torch.autograd.Variable(output_de_concat)
        varTarget = torch.autograd.Variable(target.contiguous().cuda())

        #varOutput = model(space_to_depth(varInput, 2))
        varOutput = model(varInput)
        lossvalue = loss_cls(varOutput, varTarget) 

        valLoss = valLoss + lossvalue.item() 
        varOutput = varOutput.sigmoid()

        outPRED = torch.cat((outPRED, varOutput.data), 0)
        lossValNorm += 1

    loss_list, loss_sum = weighted_log_loss(outPRED, outGT, weight=[1,1,1])

    valLoss = valLoss / lossValNorm

    auc = computeAUROC(outGT, outPRED, 3)
    auc = [round(x, 4) for x in auc]

    #accuracyscore(outGT, outPRED, c_val)

    #write in tensorboard 
    input_grid = torchvision.utils.make_grid(input) #마지막 배치 이미지만 시각화
    summary.add_image('original2_images', input_grid, epochID)

    denoised_grid = torchvision.utils.make_grid(output_de_concat) #마지막 배치 이미지만 시각화
    summary.add_image('inference_images', denoised_grid, epochID)

    
    return valLoss, auc, loss_list, loss_sum, outPRED, outGT


def train(image_size, model_name):

    # cuda visble devices 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss', 'Val Loss_Denoised', 'psnr', 'auc', 'loss']

    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    df_all = pd.read_csv(csv_path) #label 있는 파일

    kfold_path_train = '/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/fold_5_by_study/'
    kfold_path_val = '/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/fold_5_by_study/' 

    # kfold_path_train = '/home/kka0602/RSNA2019_2DNet/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/data/fold_5_by_study_image/' # 1 Kold 당 train 15,624개/val 3906개 = 19,530개 
    # kfold_path_val = '/home/kka0602/RSNA2019_2DNet/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/data/fold_5_by_study_image/' # 1 Kold 당 val 약 134,788개 
    # loss_w = [0.9, 0.8, 0.7, 0.6]
    # for w in loss_w: 
    for num_fold in range(3,4):

        print('fold_num:',num_fold)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold]) 

        f_train = open(kfold_path_train + 'fold' + str(num_fold) + '/filename_train_g.txt', 'r') 
        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/filename_val_g.txt', 'r') 
        # f_train = open('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/filename_train.txt', 'r') 
        # f_val = open('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/filename_test.txt', 'r') 

        c_train = f_train.readlines()
        c_val = f_val.readlines() 
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]   

        # #for debug
        # c_train = c_train[0:1000]
        # c_val = c_val[0:100]


        print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))

        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader3(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        model = eval(model_name+'()')
        #model = convnext_base(pretrained=True, in_22k=True)
        #model = eca_resnet101(k_size=[3, 3, 3, 3], num_classes=3, pretrained=True)
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        
        model_de = DnCNN(1, num_of_layers=8).cuda()
        model_de = torch.nn.DataParallel(model_de)

        #denoiser pretrained 
        pretrained_models = torch.load('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/src/data_test/se_resnext101_32x4d/denoising_otitis_all_pretrained/model_epoch_0_best.pth')
        model_de.load_state_dict(pretrained_models['state_dict']) 
            
        # for params in model_de.parameters():
        #     params.requires_grad = True 

        #cls pretrained 
        # pretrained_models2 = torch.load('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/src/data_test/se_resnext101_32x4d/denoising_otitis_all_pretrained/model_epoch_19_3.pth')
        # model.load_state_dict(pretrained_models2, strict=False)    

        # for params in model.parameters():
        #     params.requires_grad = True  
            
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4) #weight_decay=0.00002
        optimizer2 = torch.optim.Adam(model_de.parameters(),lr=1e-3)  #weight decay는 안넣는게 좋음,
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=1, min_lr=1e-5)

        #scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
        #scheduler = MultiStepLR(optimizer, milestones=[9, 20], gamma=0.1)

        loss_de = MSELoss().cuda()
        loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([1.0, 1.0, 1.0]).cuda())
            

        trMaxEpoch = 10       
        for epochID in range (trMaxEpoch):  #80 
            epochID = epochID + 0
            start_time = time.time()
            model.train()
            # model_de.train()    

            cls_trainLoss = 0
            total_trainLoss = 0
            lossTrainNorm = 0

            # if epochID < 10:
            #     pass
            # elif epochID < 30:
            #     if epochID != 10:
            #         scheduler = warm_restart(scheduler, T_mult=2) 
            

            for batchID, (input, target) in enumerate (train_loader): #input : noisy image
                if batchID == 0:
                    ss_time = time.time()

                print(str(batchID) + '/' + str(int(len(c_train)/train_batch_size)) + '     ' + str((time.time()-ss_time)/(batchID+1)), end='\r')

                input = input.cuda() # (16, 3, 512, 512)

                # denoiser
                masker = Masker(width=4, mode='interpolate')                   
                net_input, mask = masker.mask(input, batchID) 
                # # output_de = model_de(net_input)
                # # lossvalue2 = loss_de(output_de*mask, input*mask)      

                # # output_de = model_de(input)
                # # output_de = torch.clamp(output_de, 0.0, 1.0)
                
                #3 window denoising => concat  => aug => cls
                output_de_concat = []
                for i in range(3):
                    image = net_input[:,i:i+1,:,:]                                            
                    output_de = model_de(image)
                    output_de = torch.clamp(output_de, 0.0, 1.0) # adjcent 3 window denoising 시킨후에 concat
                    output_de_concat.append(output_de)

                output_de_concat = torch.cat(output_de_concat, dim=1)
                # lossvalue2 = loss_de(output_de_concat*mask, input*mask)      
                
                #tensorboard
                # input_grid = torchvision.utils.make_grid(input) #마지막 배치 이미지만 시각화
                # summary.add_image('original_images', input_grid, epochID)

                # denoised_ft_grid = torchvision.utils.make_grid(output_de_concat) #마지막 배치 이미지만 시각화
                # summary.add_image('denoised_ft_images', denoised_ft_grid, epochID)

                varInput = torch.autograd.Variable(output_de_concat) 
                target = target.view(-1, 3).contiguous().cuda()
                #target = target[:,:3] #첫번째 class any만 가져오기.
                varTarget = torch.autograd.Variable(target.contiguous().cuda())

                #varOutput = model(space_to_depth(varInput, 2))
                varOutput = model(varInput)

                lossvalue = loss_cls(varOutput, varTarget) # cls loss
                
                # total_loss = 0.6 * lossvalue + 0.4 * lossvalue2
                # total_trainLoss = total_trainLoss + 0.6 * lossvalue.item() + 0.4 * lossvalue2.item()
                cls_trainLoss = cls_trainLoss + lossvalue.item()

                lossTrainNorm = lossTrainNorm + 1
                
                lossvalue.backward()
                optimizer.step()
                # optimizer2.step()

                optimizer.zero_grad()
                # optimizer2.zero_grad()
        
                del lossvalue

            #total_trainLoss = total_trainLoss / lossTrainNorm    
            cls_trainLoss = cls_trainLoss / lossTrainNorm


            #summary.add_scalar('total_loss/train', total_trainLoss, epochID)
            summary.add_scalar('cls_loss/train', cls_trainLoss, epochID)

            valLoss, auc, loss_list, loss_sum, outPRED, outGT = epochVal(epochID, model, model_de, loss_cls, val_loader, c_val, val_batch_size)
        
            summary.add_scalar('cls_loss/val ', valLoss, epochID)

            summary.add_scalar('AUC_right', auc[0], epochID)
            summary.add_scalar('AUC_left', auc[1], epochID)
            summary.add_scalar('AUC_any', auc[2], epochID)
            
            #scheduler.step(valLoss)
            #add_pr_curve_tensorboard(outPRED, outGT, epochID)

            #ROC_Curve plot
            # Recall_Precision(outGT, outPRED, 3, epochID, num_fold)
            
            confusion_m = accuracyscore(outGT, outPRED, c_val)
                
            epoch_time = time.time() - start_time #1 epoch당 걸리는 time(초)

            # if (epochID+1)%10 == 0:
            #     #torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'valLoss': valLoss}, snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold)  + '.pth')
            #     torch.save({'epoch': epochID + 1, 'state_dict': model_de.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, snapshot_path + '/model_epoch_' + str(epochID) + '.pth')

            result = [
                        epochID,
                        round(optimizer.state_dict()['param_groups'][0]['lr'], 6),
                        # round(optimizer2.state_dict()['param_groups'][0]['lr'], 6),
                        # round(total_trainLoss, 5),
                        round(cls_trainLoss, 5),
                        round(valLoss, 5),
                        'confusion_m:', confusion_m,
                        'auc:', auc,
                        'loss:',loss_list,
                        loss_sum]

            print(result)

            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)  

        del model, model_de



if __name__ == '__main__':
    csv_path = '/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/data/otitis_cls_final.csv' #라벨에 맞춘 csv
    #csv_path = '/home/kka0602/RSNA2019_2DNet/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/data/stage1_train_cls.csv'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='se_resnext101_32x4d_256', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=256, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=16, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=8, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='/se_resnext101_32x4d/denoised_3bone_otitis_all', help='epoch')
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
    train(Image_size, backbone)
    # valid_snapshot(backbone, Image_size)
