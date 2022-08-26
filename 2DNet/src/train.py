import os
import time
import pandas as pd
import gc
import cv2
import csv
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data
import sys
import torch.utils.data as data
#from torchsummaryX.torchsummaryX import summary
from net.models import *

from dataset.dataset import *
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

#from torchsummaryX import summary
from tensorboardX import SummaryWriter
sumwriter = SummaryWriter()


def space_to_depth(x, block_size):
    n, c, h, w = x.size() # 32, 3, 512, 512 -> 32 ,48, 128, 128
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

def epochVal(model, dataLoader, loss_cls, c_val, val_batch_size):
    model.eval ()
    lossValNorm = 0
    valLoss = 0

    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    for i, (input, target) in enumerate (dataLoader):
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
        target = target.view(-1, 6).contiguous().cuda()
        target = target[:,:1] #첫번째 class any만 가져오기.

        outGT = torch.cat((outGT, target), 0)
        varInput = torch.autograd.Variable(input)
        varTarget = torch.autograd.Variable(target.contiguous().cuda())
        #varOutput = model(varInput)
        varOutput = model(space_to_depth(varInput, 2))
        lossvalue = loss_cls(varOutput, varTarget) 
        valLoss = valLoss + lossvalue.item() 
        varOutput = varOutput.sigmoid()

        outPRED = torch.cat((outPRED, varOutput.data), 0)
        lossValNorm += 1

    valLoss = valLoss / lossValNorm

    #rec, pre, thres = Recall_Precision(outGT, outPRED, 1)

    auc = computeAUROC(outGT, outPRED, 1)
    auc = [round(x, 4) for x in auc]

    #sumwriter.add_scalar('val loss', valLoss, i)
    #sumwriter.add_scalar('auc', auc, i)
    #f1score
    loss_list, loss_sum = weighted_log_loss(outPRED, outGT)

    return valLoss, auc, loss_list, loss_sum

def train(image_size, model_name):

    # cuda visble devices 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']

    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    df_all = pd.read_csv(csv_path)

    kfold_path_train = '../data/fold_5_by_study_image/' # 1 Kold 당 train 15,624개/val 3906개 = 19,530개 
    kfold_path_val = '../data/fold_5_by_study_image/' # 1 Kold 당 val 약 134,788개 

    for num_fold in range(5):
        print('fold_num:',num_fold)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold]) 

        f_train = open(kfold_path_train + 'fold' + str(num_fold) + '/train.txt', 'r') #15,624개 (study_instance_uid)
        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r') #134,788개 (filename)
        c_train = f_train.readlines()
        c_val = f_val.readlines() 
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]     

        #for debug
        # c_train = c_train[0:1000]
        # c_val = c_val[0:400]

        print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset:', len(c_val)])  
            writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])  

        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader2(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        model = eval(model_name+'()')
        #model = eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1, pretrained=True)

        model = model.cuda()
        print(model)

        #summary(model, torch.rand((32,9,256,256)))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
        model = torch.nn.DataParallel(model)
        loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([1.0]).cuda())
        #loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())

        #저장한 best epoch pth 불러와서 이어서 training
        # state = torch.load('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/src/data_test/se_resnext101_32x4d/pretrained_6/model_epoch_19_0.pth')
        # epoch = state['epoch']
        # best_valid_loss = state['valLoss']
        # model.load_state_dict(state['state_dict']) 

        # optimizer.load_state_dict(state['optimizer_state_dict']) 


        trMaxEpoch = 50
        for epochID in range (trMaxEpoch):  #80 
            epochID = epochID + 0

            start_time = time.time()
            model.train()
            trainLoss = 0
            lossTrainNorm = 10

            if epochID < 10:
                pass
            elif epochID < 80:
                if epochID != 10:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2) 
            else:
                optimizer.param_groups[0]['lr'] = 1e-5

            for batchID, (input, target) in enumerate (train_loader): #15624
                if batchID == 0:
                    ss_time = time.time()

                print(str(batchID) + '/' + str(int(len(c_train)/train_batch_size)) + '     ' + str((time.time()-ss_time)/(batchID+1)), end='\r')
                varInput = torch.autograd.Variable(input)
                
                target = target.view(-1, 6).contiguous().cuda()
                target = target[:,:1] #첫번째 class any만 가져오기.
                varTarget = torch.autograd.Variable(target.contiguous().cuda())
                #varOutput = model(varInput)
                varOutput = model(space_to_depth(varInput, 2))


                lossvalue = loss_cls(varOutput, varTarget)
                trainLoss = trainLoss + lossvalue.item()
                lossTrainNorm = lossTrainNorm + 1


                lossvalue.backward()
                optimizer.step()
                optimizer.zero_grad()

                #sumwriter.add_scalar('training loss', trainLoss, epochID)

                del lossvalue

            trainLoss = trainLoss / lossTrainNorm

            # if (epochID+1)%5 == 0 or epochID > 79 or epochID == 0:
            #     valLoss, auc, loss_list, loss_sum = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)

            valLoss, auc, loss_list, loss_sum, outPRED, outGT = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)
            
            

        
            epoch_time = time.time() - start_time #1 epoch당 걸리는 time(초)

            if (epochID+1)%5 == 0 or epochID > 79:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'valLoss': valLoss}, snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth')
            #torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'valLoss': valLoss}, snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth')

            result = [epochID,
                      round(optimizer.state_dict()['param_groups'][0]['lr'], 6),
                      round(epoch_time, 0),
                      round(trainLoss, 5),
                      round(valLoss, 5),
                      'auc:', auc,
                      'loss:',loss_list,
                      loss_sum]

            print(result)

            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)  

        del model

# def valid_snapshot(model_name, image_size):
#     dir = r'./DenseNet121_change_avg_256'
#     if not os.path.exists(snapshot_path):
#         os.makedirs(snapshot_path)
#     header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']

#     if not os.path.isfile(snapshot_path + '/log.csv'):
#         with open(snapshot_path + '/log.csv', 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(header)
#     df_all = pd.read_csv(csv_path)

#     kfold_path_val = '../data/rsna-intracranial-hemorrhage-detection/fold_5_by_study_image/'
#     loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())
#     for num_fold in range(5):
#         print('fold_num:', num_fold)

#         ckpt = r'model_epoch_best_'+str(num_fold)+'.pth'
#         ckpt = os.path.join(dir,ckpt)

#         with open(snapshot_path + '/log.csv', 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([num_fold])

#         f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
#         c_val = f_val.readlines()
#         f_val.close()
#         c_val = [s.replace('\n', '') for s in c_val]

#         print('  val dataset image num:', len(c_val))

#         val_transform = albumentations.Compose([
#             albumentations.Resize(image_size, image_size),
#             albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0,
#                                      p=1.0)
#         ])

#         val_dataset = RSNA_Dataset_val_window3_by_study_context(df_all, c_val, val_transform)

#         val_loader = torch.utils.data.DataLoader(
#             val_dataset,
#             batch_size=val_batch_size,
#             shuffle=False,
#             num_workers=workers,
#             pin_memory=True,
#             drop_last=False)

#         model = eval(model_name + '()')
#         model = model.cuda()
#         model = torch.nn.DataParallel(model)

#         if ckpt is not None:
#             print(ckpt)
#             model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage)["state_dict"])

#         valLoss, auc, loss_list, loss_sum = epochVal(model, val_loader, loss_cls, c_val, val_batch_size) 

#         result = [round(valLoss, 5),
#                   'auc:', auc,
#                   'loss:', loss_list,
#                   loss_sum]

#         with open(ckpt + '_log.csv', 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(result)
#         print(result)


if __name__ == '__main__':
    csv_path = '../data/stage1_train_cls.csv'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='se_resnext101_32x4d', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=256, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=32, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=16, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='se_resnext101_32x4d/', help='epoch')
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
