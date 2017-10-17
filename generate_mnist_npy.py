import cv2
import numpy as np
import glob
import os

def convert_mnist(path):
	image_list=[]
	dir_list =  [name for name in os.listdir(path)]
	for dir_name in dir_list:
		image_names = glob.glob(path+dir_name+"/*.jpg")
		# print image_names
		for image_name in image_names:
			image = 1/255.0*cv2.imread(image_name, flags=cv2.IMREAD_GRAYSCALE)
			new_image = np.zeros(shape=[28,28,1], dtype=np.float32)
			new_image[:,:,0] = image[:,:]
			image_list.append(new_image)
	image_array = np.array(image_list)
	np.random.shuffle(image_array)
	print image_array.shape
	np.save(path+"mnist_data_28*28*1_0to1values.npy", image_array)
	
path = "./dataset/trainingSet/"
convert_mnist(path)