import argparse
import os
import shutil

import paddle
from paddle.static import InputSpec
from visualdl import LogWriter, LogReader
from configs import Config, str2bool
from model import modelset
from utils import Miou, VDL, Loss, imgehance, SDataSet, SubSet


def parse_args():
    parser = argparse.ArgumentParser("Model training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cfg.yaml',
        help='path to config file of model')

    parser.add_argument(
        '--pretrain',
        default='False',
        type=str2bool,
        help='path to pretrain weights.')

    parser.add_argument(
        '--epoch',
        type = int,
        default = None,
        help = 'epoch number, 0 for read from config file')

    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help='directory name to save train snapshoot')

    args = parser.parse_args()
    return args


def main(args):
    config = Config(args.config)
    cfg = config(vars(args), mode=['train', 'init'])
    mdname = cfg['train']['model']
    vdl_dir = os.path.join(args.save_dir, cfg['init']['vdl_dir'])
    vdl_dir = os.path.join(vdl_dir, mdname)
    vdl_name = 'vdlrecords.' + mdname + '.log'
    vdl_log_dir = os.path.join(vdl_dir,vdl_name)


    fil_list = os.path.join(cfg['train']['root_path'],cfg['train']['train_list'])
    mean = cfg['train']['mean']
    std = cfg['train']['std']

    custom = cfg['train']['custom']['type']
    if custom == True:
        print('use custom data')
        mean = cfg['train']['custom']['mean']
        std = cfg['train']['custom']['std']


    # image enhance
    trfm = imgehance(size=cfg['train']['sz'])

    # load dataset
    ds = SDataSet(path=cfg['train']['root_path'],fl=fil_list,sz=cfg['train']['sz'])

    train_ds = SubSet(ds, mode='train', mean=mean, std=std, transform=trfm)
    val_ds = SubSet(ds, mode='valid', mean=mean, std=std, transform=None)

    # select model
    net = modelset(mode=mdname, num_classes=cfg['init']['num_classes'])

    # load moel
    input = InputSpec([None,3,64,64], 'float32', 'image')
    label = InputSpec([None,1,64,64], 'int64', 'label')
    model = paddle.Model(net, input, label)
    #print(model.summary((-1, 3, 64, 64)))  #
    iters = 0
    epochs = 0
    if args.pretrain:
        model.load(path=os.path.join(args.save_dir, mdname) + '/' + str(mdname))
        vdlreader = LogReader(file_path=vdl_log_dir)
        iters = vdlreader.get_data('scalar', 'train%miou')[-1].id + 1
        epochs = vdlreader.get_data('scalar', 'eval%miou')[-1].id + 1
    elif os.path.exists(vdl_dir):
        shutil.rmtree(vdl_dir)

    write = LogWriter(logdir=vdl_dir, file_name=vdl_name)

    opt = paddle.optimizer.Momentum(learning_rate=cfg['train']['lr'], parameters=model.parameters())
    model.prepare(optimizer = opt,
                  loss = Loss(),
                  metrics= Miou(num_classes=cfg['init']['num_classes'],name='miou'),
                  )

    model.fit(train_ds,
              val_ds,
              epochs=cfg['train']['epoch'],
              batch_size=cfg['train']['batchsz'],
              log_freq=1,
              save_freq = cfg['train']['save_freq'],
              save_dir=os.path.join(args.save_dir, mdname)+'/'+str(mdname),
              verbose=1,
              num_workers = cfg['train']['num_workers'],
              callbacks=VDL(write=write, iters=iters, epochs=epochs)#VDL(logdir=vdl_dir)#
              )

    print('save model in {}'.format(os.path.join(args.save_dir, mdname)))
    model.save(path=os.path.join(args.save_dir, mdname)+'/'+str(mdname))



if __name__ == '__main__':
    args = parse_args()
    if args.pretrain:
        assert os.path.exists(args.save_dir), "Error!the path does not exist or the path is not correct!"
    elif not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    main(args)


