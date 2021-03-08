import argparse

import numpy as np
import tifffile as tiff
import pandas as pd
import os

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("calculate")
    parser.add_argument(
        '--path',
        type=str,
        default="MatherCupB/label",
        help='directory name to save train snapshoot')

    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help='directory name to save train snapshoot')

    args = parser.parse_args()
    return args

def acc(path, out="output",type="tif"):

    acc_list = []
    fl_list = os.listdir(path)
    for fl in fl_list:
        file = os.path.join(path, fl)
        if type == "tif":
            img = tiff.imread(file)
        else:
            img = Image.open(file)
            img = np.array(img)
        acc = np.mean(img)
        acc_list.append(acc)

    save = dict()
    save["file"] = fl_list
    save["acc"] = acc_list
    df = pd.DataFrame(save)
    writer = pd.ExcelWriter(os.path.join(out,'result.xlsx'))
    df.to_excel(writer, sheet_name='Sheet1', index=None)
    writer.save()

if __name__ == '__main__':
    args = parse_args()
    #assert args.path == None, "Lacking img_dir "
    acc(args.path, args.save_dir,type = "png")
    print("Save file in {}".format(args.save_dir))

