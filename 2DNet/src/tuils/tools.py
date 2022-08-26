import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tensorboardX import summary
import csv

import torch
from sklearn.metrics import fbeta_score
import time
import cv2
from sklearn.metrics import log_loss
import torch
from torch.nn.functional import threshold
import matplotlib.pyplot as plt
from collections import Counter



def accuracyscore(dataGT, dataPRED, c_val): #6 class 

#     y_pred_all = []
#     y_index_pred0 = []
#     y_index_pred1 = []
    confusion_m = []
    classes =  ['right', 'left', 'any']
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(3): # (24, 2)
        # y_index_pred0 = []
        # y_index_pred1 = []
        # count_list = []

        y_pred = np.where(datanpPRED[:, i] > 0.5, 1, 0) 
  
        # for j in range(len(y_pred)):
        #     if ((datanpGT[j:j+1, i] - y_pred[j:j+1]) == 1) : # gt 1 , pred 0
        #         y_index_pred0.append(c_val[j])
        #         count_list.append(c_val[j].split('_')[-3])
        #     elif ((datanpGT[j:j+1, i] - y_pred[j:j+1]) == -1): # gt 0 , pred 1
        #         y_index_pred1.append(c_val[j])
        #         count_list.append(c_val[j].split('_')[-3])
        # result = Counter(count_list) 

        # with open('/home/kka0602/RSNA2019_2DNet/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/src/data_test/se_resnext101_32x4d/5fold_3bone_rsna_all_test/index/' + '/log.csv', 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['pred0', 'classes :', classes[i], 'index:', y_index_pred0])  
        #     writer.writerow(['pred1', 'classes :', classes[i], 'index:', y_index_pred1])  
        #     writer.writerow(['classes :', classes[i], 'max_index:', result])

        # y_pred_all.append(accuracy_score(datanpGT[:, i], y_pred)) # class 1 인 얘들의 acc , class 2인 얘들의 acc ...
        confusion_m.append(confusion_matrix(datanpGT[:, i], y_pred))

    return confusion_m

def computeAUROC(dataGT, dataPRED, classCount): #6 class 

    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount): # (24, 2)
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i])) # class 1 인 얘들의 acc , class 2인 얘들의 acc ...

    return outAUROC

def plot_roc_curve(fpr, tpr, num, num_fold):
    #classes = ['right', 'left', 'any']
    plt.plot(fpr[0], tpr[0], color='red', label='right_ROC')
    plt.plot(fpr[1], tpr[1], color='blue', label='left_ROC')
    plt.plot(fpr[2], tpr[2], color='purple', label='any_ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    #plt.show()
    plt.savefig('/home/kka0602/nas/roc_auc/3_window/' + 'fold' + str(num_fold) + '_' + str(num) + '_auc_roc_.png')
    plt.clf()



def Recall_Precision(dataGT, dataPRED, classCount, epochID, num_fold): #6 class 
    fpr_list = []
    tpr_list = []
    thres_list = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    num = epochID

    for i in range(classCount): #6

        fpr, tpr, threshold = roc_curve(datanpGT[:, i], datanpPRED[:, i])
        # f1 = search_f1(datanpGT[:, i], datanpPRED[:, i])
        #report = classification_report(datanpGT[:, i], datanpPRED[:, i])

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        thres_list.append(threshold)

    plot_roc_curve(fpr_list, tpr_list, num, num_fold) #시각화
    

    return fpr_list, tpr_list, thres_list

def search_f1(output, target):
    max_result_f1_list = []
    max_threshold_list = []
    precision_list = []
    recall_list = []
    eps=1e-20
    target = target.type(torch.cuda.ByteTensor)

    # print(output.shape, target.shape)
    for i in range(output.shape[1]):

        output_class = output[:, i]
        target_class = target[:, i]
        max_result_f1 = 0
        max_threshold = 0

        optimal_precision = 0
        optimal_recall = 0

        for threshold in [x * 0.01 for x in range(0, 100)]:

            prob = output_class > threshold
            label = target_class > 0.5
            # print(prob, label)
            TP = (prob & label).sum().float()
            TN = ((~prob) & (~label)).sum().float()
            FP = (prob & (~label)).sum().float()
            FN = ((~prob) & label).sum().float()

            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            # print(precision, recall)
            result_f1 = 2 * precision  * recall / (precision + recall + eps)

            if result_f1.item() > max_result_f1:
                # print(max_result_f1, max_threshold)
                max_result_f1 = result_f1.item()
                max_threshold = threshold

                optimal_precision = precision
                optimal_recall = recall

        max_result_f1_list.append(round(max_result_f1,3))
        max_threshold_list.append(max_threshold)
        precision_list.append(round(optimal_precision.item(),3))
        recall_list.append(round(optimal_recall.item(),3))

    return max_result_f1_list, precision_list, recall_list

def weighted_log_loss(output, target, weight=[1,1,1,1,1,1]):

    loss = torch.nn.BCELoss()
    loss_list = []
    for i in range(output.shape[1]):
        output_class = output[:, i]
        target_class = target[:, i]
        loss_class = loss(output_class, target_class)   
        loss_list.append(float(loss_class.cpu().numpy()))

    loss_sum = np.mean(np.array(weight)*np.array(loss_list))
    loss_list = [round(x, 4) for x in loss_list]
    loss_sum = round(loss_sum, 4)

    return loss_list, loss_sum 

def weighted_log_loss_numpy(output, target, weight=[1,1,1,1,1,1]):

    loss_list = []
    for i in range(output.shape[1]):
        output_class = output[:, i]
        target_class = target[:, i]
        loss_class = log_loss(target_class.ravel(), output_class.ravel(), eps=1e-7)
        loss_list.append(loss_class)

    loss_sum = np.mean(np.array(weight)*np.array(loss_list))
    loss_list = [round(x, 4) for x in loss_list]
    loss_sum = round(loss_sum, 4)

    return loss_list, loss_sum