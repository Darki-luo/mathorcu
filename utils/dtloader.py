from random import shuffle

import cv2

from paddle.io import Dataset
import tifffile as tiff
import os
import numpy as np

from paddle.vision import transforms as T

class SDataSet(Dataset):
	def __init__(self, path, fl, sz=64):
		super(SDataSet, self).__init__()
		self.path = path
		self.fl = fl
		self.sz = sz
		self.list = []
		self._get_data()
		self.len = len(self.list)

	def _get_data(self):

		with open(self.fl) as files:
			#data_list = []#np.random.random((1, self.sz, self.sz, 3))
			#lab_list = []#np.random.randint(2, size=(1, self.sz, self.sz))
			for line in files:
				image, label = line.split()
				img = tiff.imread(os.path.join(self.path, image))  # [h,w,c]
				mask = tiff.imread(os.path.join(self.path, label))
				shape = img.shape
				pad0 = self.sz - shape[0] % self.sz
				pad1 = self.sz - shape[1] % self.sz
				img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], 'constant',
				             constant_values=0)
				mask = np.pad(mask, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2]], 'constant',
				              constant_values=0)
				img = img.reshape(img.shape[0] // self.sz, self.sz, img.shape[1] // self.sz, self.sz, 3)
				img = img.transpose(0, 2, 1, 3, 4).reshape(-1, self.sz, self.sz, 3)
				mask = mask.reshape(mask.shape[0] // self.sz, self.sz, mask.shape[1] // self.sz, self.sz)
				mask = mask.transpose(0, 2, 1, 3).reshape(-1, self.sz, self.sz)
				for i in range(img.shape[0]):
					self.list.append((img[i], mask[i]))


				#lab_list.append(mask)

		#img_list = img_list[1::].astype(np.float32)
		#lab_list = lab_list[1::].astype(np.int64)
		shuffle(self.list)


	def __len__(self):

		return self.len

	def __getitem__(self, item):

		return self.list[item]


mode_list = ["train", "valid"]

class SubSet(Dataset):
	def __init__(self, dataset, mode = 'train', mean=None, std=None, transform = None):
		super(SubSet, self).__init__()
		assert mode in mode_list, "dataset get invalid mode"

		self.dataset = dataset
		self.mode = mode

		self.img_list = []
		self.mask_list = []
		self._list()
		self.len = len(self.img_list)

		self.transform = transform

		self.as_tensor = T.Compose([
			#T.ToTensor(),
			T.Normalize(mean=mean,
			            std=std,
			            ),
		])

	def _list(self):
		train_img = []
		train_mask = []
		val_img = []
		val_mask = []
		for item, dt in enumerate(self.dataset):
			#dt = self.data[item]
			if item % 16 == 0:
				val_img.append(dt[0])
				val_mask.append(dt[1])
			else:
				train_img.append(dt[0])
				train_mask.append(dt[1])

		if self.mode == 'train':
			self.img_list = train_img
			self.mask_list = train_mask

		else:
			self.img_list = val_img
			self.mask_list = val_mask

	def __len__(self):

		return self.len

	def __getitem__(self, item):
		img = self.img_list[item]
		#print(img.shape)
		mask = self.mask_list[item]
		#print(mask.shape)
		if self.transform is not None:
			augments = self.transform(image=img, mask=mask)
			img = augments['image']
		#print(img.shape)
			mask = augments['mask']
		#print(mask.shape)
		img = img.transpose(2,0,1)
		mask = mask[np.newaxis,:,:]#add new axis

		return self.as_tensor(img) / 255., np.array(mask, dtype='int64')
	
	
class CuDataset(Dataset):
	def __init__(self, path, fl, sz):
		super(CuDataset, self).__init__()
		self.path = path
		self.fl = fl
		self.sz = sz
		self.list = []
		self._get_data()
		self.len = len(self.list)

	def _get_data(self):

		with open(self.fl) as files:

			for line in files:
				image, label = line.split()
				img_path = os.path.join(self.path, image)  # [h,w,c]
				mask_path = os.path.join(self.path, label)
				self.list.append((img_path,mask_path))

		shuffle(self.list)


	def __len__(self):

		return self.len

	def __getitem__(self, item):
		img_path, label_path = self.list[item]
		img = tiff.imread(img_path)
		label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

		return (img,label)





class TeDataset(Dataset):
	def __init__(self, path, fl, sz=64):
		super(TeDataset, self).__init__()
		self.path = os.path.join(path,fl)
		self.sz = sz
		self.data = self.get_data()
		self.as_tensor = T.Compose([
			T.Normalize([90.39095958, 89.36796833, 85.25276458],
			            [3.09639721, 2.50642894, 2.7135403],
			            ),
		])

	def get_data(self):
		img_list = np.random.random((1, self.sz, self.sz, 3))
		file_list = os.listdir(self.path)
		# linux ['Test2.tif', 'Test1.tif'] ; windows: ['Test1.tif', 'Test2.tif']
		file_list.sort(key=lambda x:int(x[-5:-4]))
		# linux ['Test1.tif', 'Test2.tif'] ; windows: ['Test1.tif', 'Test2.tif']
		for image in file_list:
			img = tiff.imread(os.path.join(self.path, image))
			shape = img.shape
			pad0 = self.sz - shape[0] % self.sz
			pad1 = self.sz - shape[1] % self.sz
			img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], 'constant',
			             constant_values=0)
			img = img.reshape(img.shape[0] // self.sz, self.sz, img.shape[1] // self.sz, self.sz, 3)
			img = img.transpose(0, 2, 1, 3, 4).reshape(-1, self.sz, self.sz, 3)
			img_list = np.append(img_list, img, axis=0)

		img_list = img_list[1::].astype(np.float32)
		return img_list



	def __len__(self):
		return len(self.data)


	def __getitem__(self, item):
		img = self.data[item]
		img = img.transpose(2,0,1)
		return self.as_tensor(img) / 255.


