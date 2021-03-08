import paddle
from paddle.nn import functional as F
from paddle.nn import Layer
from paddleseg.models.losses import BCELoss, DiceLoss

class Loss1(BCELoss):
	def __init__(self,
                 weight=None,
                 pos_weight=None,
                 ignore_index=255,
                 edge_label=False):
		super(Loss1, self).__init__(
                 weight=weight,
                 pos_weight=pos_weight,
                 ignore_index=255,
                 edge_label=ignore_index)

	def forward(self, logit, label):
		eps = 1e-6
		if len(label.shape) != len(logit.shape):
			label = paddle.unsqueeze(label, 1)
		# label.shape should equal to the logit.shape
		if label.shape[1] != logit.shape[1]:
			label = label.squeeze(1)
			label = F.one_hot(label, logit.shape[1])
			label = label.transpose((0, 3, 1, 2))
		mask = (label != self.ignore_index)
		mask = paddle.cast(mask, 'float32')
		if isinstance(self.weight, str):
			pos_index = (label == 1)
			neg_index = (label == 0)
			pos_num = paddle.sum(pos_index.astype('float32'))
			neg_num = paddle.sum(neg_index.astype('float32'))
			sum_num = pos_num + neg_num
			weight_pos = 2 * neg_num / (sum_num + eps)
			weight_neg = 2 * pos_num / (sum_num + eps)
			self.weight = weight_pos * label + weight_neg * (1 - label)
		if isinstance(self.pos_weight, str):
			pos_index = (label == 1)
			neg_index = (label == 0)
			pos_num = paddle.sum(pos_index.astype('float32'))
			neg_num = paddle.sum(neg_index.astype('float32'))
			sum_num = pos_num + neg_num
			self.pos_weight = 2 * neg_num / (sum_num + eps)
		label = label.astype('float32')
		bce_loss = F.binary_cross_entropy_with_logits(
			logit,
			label,
			weight=self.weight,
			reduction='none',
			pos_weight=self.pos_weight)

		dice_loss = F.dice_loss(
			logit,
			label,
			epsilon=eps
		)
		loss = (bce_loss + dice_loss) / 2.
		loss = loss * mask
		loss = paddle.mean(loss) / paddle.mean(mask + eps)
		label.stop_gradient = True
		mask.stop_gradient = True

		return loss




class Loss(Layer):
	def __init__(self):
		super(Loss, self).__init__()
		self.bce = BCELoss()
		self.dice = DiceLoss()

	def forward(self, logit, label):
		bce_loss = self.bce(logit, label)
		dice_loss = self.dice(logit, label)

		loss = (bce_loss + dice_loss) / 2.

		return loss





