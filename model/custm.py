import paddle
from paddle.nn import Layer, Dropout, Conv2D
from paddle.nn import functional as F
from paddleseg import models as M
import numpy as np
#from paddleseg.models.layers import ConvBNReLU



class Gcnhead(Layer):
	def __init__(self,
	             num_classes,
	             backbone_indices,
	             backbone_channels,
	             gc_channels,
	             ratio):

		super(Gcnhead, self).__init__()

		in_channels = backbone_channels[0]
		self.conv_bn_relu1 = M.layers.ConvBNReLU(
			in_channels=in_channels,
			out_channels=gc_channels,
			kernel_size=3,
			padding=1)

		self.gc_block = M.GlobalContextBlock(in_channels=gc_channels, ratio=ratio)

		self.conv_bn_relu2 = M.layers.ConvBNReLU(
			in_channels=gc_channels,
			out_channels=gc_channels,
			kernel_size=3,
			padding=1)

		self.conv_bn_relu3 = M.layers.ConvBNReLU(
			in_channels=in_channels + gc_channels,
			out_channels=gc_channels,
			kernel_size=3,
			padding=1)

		self.dropout = Dropout(p=0.1)

		self.conv = Conv2D(
			in_channels=gc_channels, out_channels=num_classes, kernel_size=1)

		self.backbone_indices = backbone_indices

	def forward(self, feat_list):
		logit_list = []
		x = feat_list[self.backbone_indices[0]]

		output = self.conv_bn_relu1(x)
		output = self.gc_block(output)
		output = self.conv_bn_relu2(output)

		output = paddle.concat([x, output], axis=1)
		output = self.conv_bn_relu3(output)

		output = self.dropout(output)
		logit = self.conv(output)
		logit_list.append(logit)

		return logit_list


class Gc(Layer):
	def __init__(self,
	             num_classes,
                 backbone,
                 backbone_indices=(-1,),
                 gc_channels=256,
                 ratio=0.25,
                 align_corners=False,
                 ):
		super(Gc, self).__init__()
		self.backbone = backbone
		backbone_channels = [
			backbone.feat_channels[i] for i in backbone_indices
		]
		#print(len(backbone_channels))
		self.head = Gcnhead(num_classes, backbone_indices, backbone_channels,
		                      gc_channels, ratio)
		self.align_corners = align_corners


	def forward(self, x):
		feat_list = self.backbone(x)
		#print(len(feat_list))
		logit_list = self.head(feat_list)
		#print(len(feat_list))
		return [
			F.interpolate(
				logit,
				x.shape[2:],
				mode='bilinear',
				align_corners=self.align_corners) for logit in logit_list
		]

class Sense(Layer):
	def __init__(self,
	             num_classes,
	             backbone,
	             backbone_indices=(-1,),
	             align_corners=False,
	             ):
		super(Sense, self).__init__()

		self.backbone = backbone
		backbone_channels = [
			backbone.feat_channels[i] for i in backbone_indices
		]

		self.fcnhead = M.FCNHead(num_classes, backbone_indices, backbone_channels,
		                    channels=None)

		self.gcnhead = Gcnhead(num_classes, backbone_indices, backbone_channels,
		                      gc_channels=256, ratio=0.25)

		self.classfier = Conv2D(num_classes*2, num_classes,kernel_size=1)


		self.align_corners = align_corners

	def forward(self, x):
		feat_list = self.backbone(x)
		fcnlogits = self.fcnhead(feat_list)[0]
		gcnlogits = self.gcnhead(feat_list)[0]
		#print(fcnlogits[0].shape)
		#print(gcnlogits[0].shape)
		logits = paddle.concat([fcnlogits,gcnlogits],axis=1)

		logits = self.classfier(logits)

		#print(logits.shape)
		return F.interpolate(logits,x.shape[2:],mode='bilinear',align_corners=self.align_corners)


if __name__ == '__main__':

	network = Sense(num_classes=2,  backbone=M.HRNet_W18(pretrained=None), align_corners=False)
	#model = Gc(num_classes=2,  backbone=M.HRNet_W18(pretrained=None), align_corners=False)
	# x =  np.random.rand(2, 3, 64, 64).astype(np.float32)#.astype('float64')
	# #print(x.shape)
	# x = paddle.to_tensor(x)
	# pred = network(x)

	model = paddle.Model(network)

	print(model.summary((-1, 3, 64, 64)))
