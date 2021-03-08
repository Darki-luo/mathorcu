import argparse
import os

import paddle
from paddle.static import InputSpec
import numpy as np

from visualdl import LogWriter
from configs.cfg import Config
from model import modelset
from utils import TeDataset, construct, saveimg, colorize, blend_image, create_path
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser("Image infer")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cfg.yaml',
        help='path to config file of model')


    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help='directory name to save train snapshoot')

    args = parser.parse_args()
    return args


def main(args):
    config = Config(args.config)
    cfg = config(vars(args), mode=['infer', 'init'])
    scale = cfg['infer']['scale']
    mdname = cfg['infer']['model']
    imgname = ''.join(mdname) # + '/' + str(scale)
    #dirname = ''.join(mdname)  + '_' + str(scale)
    sz = cfg['infer']['sz']
    infer_size = cfg['infer']['infer_size']
    #save_path = os.path.join(args.save_dir, cfg['init']['result'])
    list = cfg['infer']['infer_size']
    save_path = create_path(args.save_dir, cfg['init']['result'])
    save_path = create_path(save_path, imgname)
    save_path = create_path(save_path, str(scale))

    tif_path = create_path(save_path, cfg['infer']['lab'])
    color_path = create_path(save_path, cfg['infer']['color'])
    gray_path = create_path(save_path,cfg['infer']['gray'])
    vdl_dir = os.path.join(args.save_dir, cfg['init']['vdl_dir'])
    palette = cfg['infer']['palette']
    palette=np.array(palette,dtype=np.uint8)
    num_class = cfg['init']['num_classes']
    batchsz = cfg['infer']['batchsz']
    infer_path = os.path.join(cfg['infer']['root_path'],cfg['infer']['path'])
    tagname = imgname + '/' + str(scale)
    vdl_dir = os.path.join(vdl_dir, 'infer')
    writer = LogWriter(logdir=vdl_dir)

    infer_ds = TeDataset(path=cfg['infer']['root_path'],
                               fl=cfg['infer']['path'],
                               sz=sz)
    total = len(infer_ds)
    # select model
    #addresult = np.zeros((total//batchsz,batchsz,num_class,sz,sz))
    addresult = np.zeros((total,num_class,sz,sz))
    for mnet in mdname:
        result_list = []
        net = modelset(mode=mnet, num_classes=cfg['init']['num_classes'])
        # load moel
        input = InputSpec([None, 3, 64, 64], 'float32', 'x')
        label = InputSpec([None, 1, 64, 64], 'int64', 'label')
        model = paddle.Model(net, input, label)
        model.load(path=os.path.join(args.save_dir, mnet) + '/' + mnet)
        model.prepare()
        result = model.predict(infer_ds,
                               batch_size=batchsz,
                               num_workers=cfg['infer']['num_workers'],
                               stack_outputs = True # [160,2,64,64]
                               )

        addresult = result[0] + scale * addresult

    pred = construct(addresult, infer_size, sz = sz)
    # pred = construct(addresult,infer_size,sz = sz)
    # # 腐蚀膨胀
    # read vdl
    file_list = os.listdir(infer_path)
    file_list.sort(key=lambda x: int(x[-5:-4]))
    step = 0
    for i,fl in enumerate(file_list):
        name,_= fl.split(".")
        # save pred
        lab_img = Image.fromarray(pred[i].astype(np.uint8)).convert("L")
        saveimg(lab_img, tif_path, name=name, type='.tif')

        # gray_label
        label = colorize(pred[i], palette)
        writer.add_image(tag=tagname,
                         img=saveimg(label, gray_path, name=name, type='.png', re_out=True),
                         step=step,
                         dataformats='HW')
        step += 1

        # color_label
        file = os.path.join(infer_path,fl)
        out = blend_image(file, label, alpha=0.25)
        writer.add_image(tag=tagname,
                         img=saveimg(out, color_path, name=name, type='.png', re_out=True),
                         step=step,
                         dataformats='HWC')
        step += 1
    writer.close()










if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.save_dir), "Error!the path does not exist or the path is not correct!"
    main(args)