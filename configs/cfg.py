import logging

import yaml
import codecs


def log_info(filename):
    logging.root.handlers = []
    FORMAT = '[ %(asctime)s ] %(message)s'
    datemt='%b,%d %H:%M:%S'
    logging.basicConfig(filename=filename, level=logging.INFO, format=FORMAT,datefmt=datemt)
    logger = logging.getLogger(__name__)

    return logger

def log_init(logger, dic, i=0):
    j=i
    for key, value in dic.items():
        if isinstance(value,dict):
            space1 = "-" * i
            logging.info("{}{}:".format(space1, key))
            j += 1
            log_init(logger, value, i=j)
            j = i
        else:
            space2="-" * j
            logger.info("{}{}: {}".format(space2,key,value))



mode = ['train','val','infer','init','model']

def update_config(dic, arg_dict, mode=None):
    up_dic = arg_dict.copy()
    sec_dic = dic.copy()

    if mode is not None:
        base_dic = sec_dic[mode]
    else:
        base_dic = sec_dic

    for key, value in base_dic.items():
        if isinstance(value, dict):
            base_dic[key] = update_config(base_dic[key],up_dic)
        elif key in up_dic and up_dic[key] is not None:
            base_dic[key]=up_dic[key]
        else:
            pass

    if mode is not None:
        sec_dic[mode] = base_dic
    else:
        sec_dic = base_dic

    return sec_dic
    #set_dic = getattr(dic, mode)


def parse_from_yaml(path, arg_dict, mode=None):
    with codecs.open(path, 'r', 'utf-8') as file:
    #with codecs.open(path) as file:
        dic = yaml.load(file, Loader=yaml.FullLoader) #load: yaml -> dict
        #dic = update_config(dic,arg_dict,mode=mode)

    return dic

def str2bool(str):
    return True if str.lower() == 'true' else False

class Config(object):
    def __init__(self, path):

        self.path = path
        self.dic = self.parse_from_yaml()

    def parse_from_yaml(self):
        file = codecs.open(self.path, 'r', 'utf-8')
        dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    def _update_config(self, dic, arg_dict, mode=None):
        up_dic = arg_dict.copy()
        sec_dic = dic.copy()

        if mode is not None:
            base_dic = sec_dic[mode]
        else:
            base_dic = sec_dic

        for key, value in base_dic.items():
            if isinstance(value, dict):
                base_dic[key] = update_config(base_dic[key], up_dic)
            elif key in up_dic and up_dic[key] is not None:
                base_dic[key] = up_dic[key]
            else:
                pass

        if mode is not None:
            sec_dic[mode] = base_dic
        else:
            sec_dic = base_dic

        return sec_dic

    def __call__(self, args, mode=None):
        lists = ['train', 'val', 'infer', 'init', 'model']
        dic = self.dic
        if isinstance(mode, list):
            for md in mode:
                assert md in lists, "Error, mode is invalid!"
                dic=self._update_config(dic,args,md)
        else:
            assert mode in lists, "Error, mode is invalid!"
            dic = self._update_config(dic, args, mode)
        return dic







