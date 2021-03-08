from albumentations import *


class Transform(object):
	def __init__(self, size, p=0.5):
		super(Transform, self).__init__()
		self.compose = Compose([
			Resize(size,size),
			HorizontalFlip(p=p),
			VerticalFlip(p=p),

			OneOf([
				RandomBrightnessContrast(),
			],p=p),
			OneOf([
				GaussianBlur(blur_limit=7, always_apply=False, p=p)
			], p=p)
		])

def imgehance(size, p=0.5):
	trfm = Compose([
			Resize(size,size),
			HorizontalFlip(p=p),
			VerticalFlip(p=p),

			OneOf([
				RandomBrightnessContrast(),
			],p=p),
			OneOf([
				GaussianBlur(blur_limit=7, always_apply=False, p=p)
			], p=p)
		])

	return trfm



