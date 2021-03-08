from model import SateNet, Sense
from paddleseg import models as M

net_list = ['unet','att-unet','fcn','ocr','gcn','psp', 'unet++','sate']


# select model 
def modelset(mode, num_classes=2):
	assert mode in net_list, "model does not exist!,please choice in {}".format(net_list)
	if mode == 'unet':
		#net = M.UNet(num_classes=num_classes, pretrained=)
		net = SateNet(num_classes=num_classes)
	elif mode == 'att-unet':
		net = M.AttentionUNet(num_classes=num_classes)
	elif mode == 'unet++':
		net = M.UNetPlusPlus(in_channels=3,num_classes=num_classes)
	elif mode == 'psp':
		pretrained = 'https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz'
		net = M.PSPNet(num_classes=num_classes, backbone=M.ResNet50_vd(pretrained=pretrained), enable_auxiliary_loss=False)
	elif mode == 'ocr':
		pretrained = 'https://bj.bcebos.com/paddleseg/dygraph/hrnet_w18_ssld.tar.gz'
		net = M.OCRNet(num_classes=num_classes, backbone=M.HRNet_W18(pretrained=pretrained, align_corners=False), backbone_indices=(-1, ))
	elif mode == 'fcn':
		pretrained = 'https://bj.bcebos.com/paddleseg/dygraph/hrnet_w18_ssld.tar.gz'
		net = M.FCN(num_classes=num_classes, backbone=M.HRNet_W18(pretrained=pretrained), align_corners=False)
	elif mode == 'gcn':
		pretrained = 'https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz'
		#pretrained = 'https://bj.bcebos.com/paddleseg/dygraph/resnet101_vd_ssld.tar.gz'
		net = M.GCNet(num_classes=num_classes, backbone=M.ResNet50_vd(pretrained=pretrained), align_corners=False, enable_auxiliary_loss=False)
	elif mode == 'sate':
		pretrained = 'https://bj.bcebos.com/paddleseg/dygraph/hrnet_w18_ssld.tar.gz'
		net = Sense(num_classes=num_classes, backbone=M.HRNet_W18(pretrained=pretrained), align_corners=False)
	return net


