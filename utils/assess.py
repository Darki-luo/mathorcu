import paddle
import paddleseg
from paddle.metric import Metric
import numpy as np

from paddleseg.utils.metrics import calculate_area

# assess mean iou
class Miou(Metric):
    def __init__(self, num_classes, name=None):
        super(Miou, self).__init__()
        self.num_classes = num_classes
        self._name = name
        self.cla_iou = []
        self.miou = 0


    def name(self):

        return self._name

    def compute(self,  pred, label, *kwargs):
        pred = paddle.argmax(pred, axis=1)
        intersect_area, pred_area, label_area = calculate_area(pred, label, self.num_classes)
        return intersect_area, pred_area, label_area

    def update(self, intersect_area, pred_area, label_area, *kwargs):

        union = pred_area + label_area - intersect_area
        #class_iou = []
        for i in range(len(intersect_area)):
            if union[i] == 0:
                iou = 0
            else:
                iou = intersect_area[i] / union[i]
            self.cla_iou.append(iou)
        self.miou = np.mean(self.cla_iou)

        #,  = mean_iou(intersect_area, pred_area, label_area)
        return self.miou

    def accumulate(self):

        return self.miou

    def reset(self):
        """
        do reset action
        """
        self.cla_iou = []
        self.miou = 0

