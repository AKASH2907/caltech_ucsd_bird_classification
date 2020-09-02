import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import glob
import os
from os.path import isfile, join, split
from os import rename, listdir, rename, makedirs
from random import shuffle
import glob
import pickle
import matplotlib.pyplot as plt

train_dir = listdir("train")
images = []

for sub_dir in train_dir:
	images += glob.glob(join("train", sub_dir, '*.jpg\n'))
	
# Train_data and train_label array
train_data = []
train_label = []
count = 0
for j in images:

	img = cv2.imread(j)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (224, 224))
	train_data.append(img)
	train_label+=[j.split('/')[-2]]

	if count%1000==0:
		print("Number of images done:", count)
	count+=1

# print(train_label)

train_data = np.array(train_data).astype("float32")
train_label = np.array(train_label)

print(train_data.shape, train_label.shape)
np.save("train_data.npy", train_data)
np.save("train_label.npy", train_label)
