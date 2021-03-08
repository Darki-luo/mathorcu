import cv2
import numpy as np
import tifffile as tiff
import os

# def get_create(root,fl,sz=64):
# 	mean = []
# 	std = []
# 	with open(fl) as files:
# 		img_list = np.random.random((1,sz,sz,3))
# 		lab_list = np.random.randint(2,size=(1,sz,sz))
# 		for line in files:
# 			image, label = line.split()
# 			img = tiff.imread(os.path.join(root,image)) # [h,w,c]
# 			mean.append(np.mean(img.reshape(-1,3),axis=0))
# 			std.append(np.std(img.reshape(-1,3),axis=0, ddof=1))
# 			mask = tiff.imread(os.path.join(root,label))
# 			shape = img.shape
# 			pad0 = sz - shape[0] % sz
# 			pad1 = sz - shape[1] % sz
# 			img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], 'constant',
# 			             constant_values=0)
# 			mask = np.pad(mask, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2]], 'constant',
# 			              constant_values=0)
# 			img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
# 			img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
# 			mask = mask.reshape(mask.shape[0] // sz, sz, mask.shape[1] // sz, sz)
# 			mask = mask.transpose(0, 2, 1, 3).reshape(-1, sz, sz)
# 			img_list = np.append(img_list,img,axis=0)
# 			lab_list = np.append(lab_list,mask,axis=0)
#
#
# 	img_list = img_list[1::]
# 	lab_list = lab_list[1::]
# 	mean = np.mean(mean, axis=0)
# 	std = np.std(std,axis=0, ddof=1)
# 	return (img_list, lab_list), mean, std

def get_create(root, fl):
	mean = []
	std = []
	with open(fl) as files:
		for line in files:
			image = line.split()[0]
			img = tiff.imread(os.path.join(root, image))
			#img = cv2.imread(os.path.join(root, image), cv2.IMREAD_COLOR)
			#print(img.shape)
			#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			mean.append(np.mean(img.reshape(-1, 3), axis=0))
			std.append(np.std(img.reshape(-1, 3), axis=0, ddof=1))


	mean = np.mean(mean, axis=0)
	std = np.std(std,axis=0, ddof=1)
	return  mean, std







if __name__ == '__main__':
	#path = 'dataset'
	path = 'dataset/custom'
	#fl = 'dataset/train_list.txt'
	fl = 'dataset/custom/train_list.txt'
	mean,std = get_create(path, fl)
	#print(len(data))
	print(mean)
	print(std)
	#img, mask = data
	#print(img.shape)
	#print(len(img))
	#print(mask.shape)









