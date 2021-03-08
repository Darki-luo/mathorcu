import argparse
import os

import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
import pandas as pd





def make_one_hot(data1):
    return (np.arange(2)==data1[:,None]).astype(np.integer)

def TFNP(pred,label,num_classes=2):
    pred = pred.reshape(-1, 1)
    pred = make_one_hot(pred)
    label = label.reshape(-1,1)
    label = make_one_hot(label)
    xor = pred * label
    T=[]
    for i in range(num_classes):
        xor_i = xor[:,:, i]
        xor_area_i = np.sum(xor_i)
        T.append(xor_area_i)
    #1为TN,2为TP
    return T

def accmap(label_path,pred_path,num_classes=2):
    TNP =[]
    for file in os.listdir(label_path):
        file_path = os.path.join(label_path,file)
        label = tiff.imread(file_path)
        p_path = os.path.join(pred_path, file.replace(".tif", ".png"))
        #p_path = os.path.join(pred_path,file.replace("_reference.tif",".png"))
        pred = Image.open(p_path)
        pred = np.array(pred)
        T = TFNP(pred, label, num_classes=num_classes)
        TNP.append(T)
    return TNP

def pred_num(pred_path):
    pred_num = []
    for file in os.listdir(pred_path):
        p_path = os.path.join(pred_path, file)
        pred = Image.open(p_path)
        pred = np.array(pred)
        pred = np.sum(pred)
        pred_num.append(pred)
    return pred_num





if __name__ == '__main__':
    label_path = "E:\Python\Code\mathcup\output\json"
    #label_path = "E:\Python\Code\mathcup\dataset\label"
    network = "fcnunet++"
    pred_path = os.path.join("E:\\Python\\Code\\mathcup\\output\\result",network)
    #"E:\\Python\\Code\\mathcup\\output\\result\\unet\\1\\gray"
    pred_path = os.path.join(pred_path,"10\\gray")
    fl_list = os.listdir(pred_path)
    T = accmap(label_path, pred_path, num_classes=2)
    pred_nums = pred_num(pred_path)
    lab_nums = pred_num(label_path)
    precisions = []
    recalls = []
    networks = []
    print(T)
    for i in range(len(T)):
        precision = (T[i][1])/pred_nums[i]
        recall = (T[i][1])/lab_nums[i]
        precisions.append(precision)
        recalls.append(recall)
        networks.append(network)
    print(fl_list)
    print(precisions)
    print(recalls)
    print(networks)
    save = dict()
    save["network"] = networks
    save["lab"] = fl_list
    save["precision"] = precisions
    save["recall"] = recalls


    df = pd.DataFrame(save)
    writer = pd.ExcelWriter(os.path.join('output', 'result.xlsx'))
    df.to_excel(writer, sheet_name='Sheet1', index=None)
    writer.save()

