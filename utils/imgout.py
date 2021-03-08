import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import paddle
import numpy as np
from PIL import Image



def create_path(root, path):
    path = os.path.join(root, path)
    if not os.path.exists(path):
        os.mkdir(path)
        #shutil.rmtree(path)
    return path

def construct(pred, infer_size, sz = 64):
    pred = np.argmax(pred, axis=-3)
    pad0 = sz - infer_size[0] % sz
    pad1 = sz - infer_size[1] % sz
    h_scale = (pad0 + infer_size[0]) // sz
    w_scale = (pad1 + infer_size[1]) // sz
    pred = pred.reshape(-1, h_scale, w_scale, sz, sz).transpose(0, 1, 3, 2, 4)
    pred = pred.reshape(-1, h_scale * sz, w_scale * sz)
    pred = pred[:,pad0 // 2:infer_size[0] + pad0 // 2, pad1 // 2:infer_size[1] + pad1 // 2]
    return pred


def construct1(result,infer_size,sz=64):
    """
    infer_size :
    """
    results = result.transpose((0, 1, 3, 4, 2)) #[n,b,h1,w1,c]
    mask = np.argmax(results, axis=-1)
    mask1 = mask.reshape(-1, sz, sz)
    pad0 = sz - infer_size[0] % sz
    pad1 = sz - infer_size[1] % sz
    number1 = (pad0+infer_size[0])//sz
    number2 = (pad1+infer_size[1])//sz
    mask2 = mask1.reshape(-1, number1, number2, sz, sz).transpose(0, 1, 3, 2, 4)
    pred_list = []
    for data in mask2:
        maskp = data.reshape(pad0+infer_size[0], pad1+infer_size[1])
        maskm = maskp[pad0 // 2:infer_size[0] + pad0 // 2, pad1 // 2:infer_size[1] + pad1 // 2]
        pred_list.append(maskm)
    return np.array(pred_list)




    # for i in mask2:
    #     maskp = i.reshape(pad0+list[0], pad1+list[1])
    #     maskm = maskp[pad0//2:list[0]+pad0//2, pad1//2:list[1]+pad1//2]
    #     #maskm = maskm.astype('uint8')
    #     numpy_array.append(maskm)





typename = ['.png','.tif']
def saveimg(img,path,name,type='.png',re_out=False):
    assert type in typename,"error"
    filename = os.path.join(path,name+type)
    img.save(filename)
    if re_out is True:
        return np.array(img)


