# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import cv2
import numpy as np
import glob
import os

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dictionery = cPickle.load(fo)
    return dictionery
def gen_npy(batch_names):
	all_images = []
	for file in batch_names:
		dictionery = unpickle(file)
		image =  dictionery['data']
		all_images.append(image)
	all_images = tuple(all_images)
	all_images_arr = np.vstack(all_images)
	np.save("./cifar10_data/cifar_training.npy", all_images_arr)

path = "./cifar10_data" 
names = glob.glob(path+"/*")
batch_names = [name for name in names if 'data_batch' in name]
# print batch_names
gen_npy(batch_names)
