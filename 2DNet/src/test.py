import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
#from sklearn.metrics.ranking import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data

import torch.utils.data as data
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
        outGT = torch.cat((outGT, target), 0)
        varInput = torch.autograd.Variable(input)
        varTarget = torch.autograd.Variable(target.contiguous().cuda())
        varOutput = model(varInput)
        lossvalue = loss_cls(varOutput, varTarget)
        valLoss = valLoss + lossvalue.item()
        varOutput = varOutput.sigmoid()
        print(varOutput.shape)
        print(outPRED.shape)

        outPRED = torch.cat((outPRED, varOutput.data), 0)
        print(outPRED)
        lossValNorm += 1

    valLoss = valLoss / lossValNorm

    rec, pre = Recall_Precision(outGT, outPRED, 6)
    rec = [round(x, 4) for x in rec]
    pre = [round(x, 4) for x in pre]

    auc = computeAUROC(outGT, outPRED, 6)
    auc = [round(x, 4) for x in auc]
    loss_list, loss_sum = weighted_log_loss(outPRED, outGT)

    return valLoss, rec, pre, auc, loss_list, loss_sum

def train(model_name, image_size):

    # cuda visble devices 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_batch_size = 16   
    df_all = pd.read_csv(csv_path)

    kfold_path_train = '../data/fold_5_by_study/'
    kfold_path_val = '../data/fold_5_by_study_image/'

    for num_fold in range(5):
    
        print('fold_num:',num_fold)

        f_train = open(kfold_path_train + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        print('전체 train set 갯수 : {0}'.format(len(c_train)))
        c_val = f_val.readlines()
        print('전체 val set 갯수 : {0}'.format(len(c_val)))
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]     

        # for debug
        c_train = c_train[0:1000] #100장
        c_val = c_val[0:4000] #400장

        print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))

        train_transform, val_transform = generate_transforms(image_size) #aug로 입력한 image size

        # concat 3 train data 생성
        train_loader, val_loader = generate_dataset_loader2(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers) 

        model = eval(model_name+'()')
        model = model.cuda()

        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
        #scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
        model = torch.nn.DataParallel(model)

        # 저장한 best epoch pth 불러오기 
        state = torch.load('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/src/DenseNet121_change_avg_256/' + 'model_epoch_best_{0}.pth'.format(num_fold))
        epoch = state['epoch']
        best_valid_loss = state['valLoss']
        model.load_state_dict(state['state_dict'])

        loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())
        
        valLoss, rec, pre, auc, loss_list, loss_sum = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)
        result = [    round(valLoss, 5),
                      'auc:', auc,
                      'recall:', rec,
                      'precision:', pre,
                      'loss:',loss_list,
                      loss_sum]

        print(result)

        del model



if __name__ == '__main__':
    csv_path = '../data/stage1_train_cls.csv'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='DenseNet121_change_avg', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=256, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=32, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=32, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='DenseNet169_change_avg', help='epoch')  # 모델 저장.
    args = parser.parse_args()

    Image_size = args.Image_size
    #train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size #16
    workers = 24 # 24
    backbone = args.backbone
    print(backbone)
    print('image size:', Image_size)
    #print('train batch size:', train_batch_size)
    print('val batch size:', val_batch_size)
    #snapshot_path = 'data_test/' + args.model_save_path.replace('\n', '').replace('\r', '')
    train(backbone, Image_size)
    # valid_snapshot(backbone, Image_size)
