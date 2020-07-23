from keras.models import Model
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception
from keras.applications import xception
import numpy as np
from keras.preprocessing import image
from keras import utils
from PIL import Image
import cv2
from os import listdir, makedirs
from os.path import join, exists
from sklearn.preprocessing import LabelEncoder
import random
import efficientnet.keras as efn
import argparse
import keras.backend as K
import sys

def cnn_model(model_name, img_size):
	"""
	Model definition using Xception net architecture
	"""
	input_size = (img_size, img_size, 3)

	if model_name == "xception":
		baseModel = Xception(
			weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
		)
	elif model_name == "efn0":
		baseModel = efn.EfficientNetB0(weights="imagenet", include_top=False,
			input_shape=input_size)
	elif model_name == "efn_noisy":
		baseModel = efn.EfficientNetB5(weights="noisy-student", include_top=False,
			input_shape=input_size)

	headModel = baseModel.output
	headModel = GlobalAveragePooling2D()(headModel)
	headModel = Dense(1024, activation="relu", kernel_initializer="he_uniform")(
		headModel
	)
	headModel = Dropout(0.4)(headModel)
	predictions = Dense(
		200,
		activation="softmax",
		kernel_initializer="he_uniform")(
		headModel
	)
	model = Model(inputs=baseModel.input, outputs=predictions)

	for layer in baseModel.layers:
		layer.trainable = False

	optimizer = Nadam(
		lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
	)
	model.compile(
		# loss="categorical_crossentropy",
		loss=joint_loss,
		optimizer=optimizer,
		metrics=["accuracy"]
	)
	return model


def categorical_focal_loss_fixed(y_true, y_pred, gamma, alpha):
	"""
	:param y_true: A tensor of the same shape as `y_pred`
	:param y_pred: A tensor resulting from a softmax
	:return: Output tensor.
	"""

	# Scale predictions so that the class probas of each sample sum to 1
	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

	# Clip the prediction value to prevent NaN's and Inf's
	epsilon = K.epsilon()
	y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

	# Calculate Cross Entropy
	cross_entropy = -y_true * K.log(y_pred)

	# Calculate Focal Loss
	loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

	# Compute mean loss in mini_batch
	return K.mean(loss, axis=1)

	# return categorical_focal_loss_fixed


def cat_loss(y_true, y_pred):
	return K.categorical_crossentropy(y_true, y_pred)


def joint_loss(y_true, y_pred):
	# mse_loss = K.mean(K.square(y_true - y_pred))
	foc_loss = categorical_focal_loss_fixed(y_true, y_pred, alpha=.25, gamma=2.)
	cat_loss = K.categorical_crossentropy(y_true, y_pred)
	return foc_loss + cat_loss


def main():
	# train_data_mean -> shape (299, 299, 3)
	train_data_mean = np.load("/var/www/html/birds/akash_image_code/train_data_mean.npy")

	labels = np.load("/var/www/html/birds/akash_image_code/train_label.npy")
	lb = LabelEncoder()
	onehot = lb.fit_transform(labels)

	# Loading model weights
	model = cnn_model(model_name="efn_noisy", img_size=299)
	model.load_weights("/var/www/html/birds/akash_image_code/mixup_299_joint.hdf5")
	
	img_path = sys.argv[1]
	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x /= 255
	x -= train_data_mean

	predictions = model.predict(x)
	top_5 = predictions[0]
	top_5 = top_5.argsort()[-5:][::-1]
	# print(lb.inverse_transform(top_5)[0])
	y_predictions = list()
	# predict = lb.inverse_transform([np.argmax(predictions)])[0]
	# predict = predict.split(".")[1]
	# s = ''.join(predict.split("_"))
	# y_predictions = list()
	# # y_predictions.append(lb.inverse_transform([y_pred])[0])
	# y_predictions.append(s)
	# print(y_predictions)
	for i in range(5):
		predict = lb.inverse_transform(top_5)[i]
		predict = predict.split(".")[1]
		s = predict.split("_")
		a = list()
		for i in range(0, len(s)):
			a.append(s[i].title())
		y="_".join(a)
		y_predictions.append(y)
		# print(y_predictions)
	
	with open( "birds_class.txt", 'w') as f:
		for i in range(5):
			probab = 100*float(predictions[0][top_5[i]])
			# f.write("%.3f:%s\n" % (probab, lb.inverse_transform(top_5)[i]))
			f.write("%.3f:%s\n" % (probab, y_predictions[i]))
			# f.write("%s\n" % lb.inverse_transform(top_5)[i])


if __name__ == "__main__":
	main()
