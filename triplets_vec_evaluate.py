import argparse
import pickle

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam, Nadam
from keras import backend as K
# from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

## for visualizing 
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import efficientnet.keras as efn
from sklearn.preprocessing import LabelEncoder
import sys
from keras.preprocessing import image


def cnn_model(model_name, img_size, weights):
	"""
	Model definition using Xception net architecture
	"""
	input_size = (img_size, img_size, 3)

	if model_name == "efn0":
		baseModel = efn.EfficientNetB0(weights="imagenet", include_top=False,
			input_shape=input_size)
	elif model_name == "efn_noisy":
		baseModel = efn.EfficientNetB5(weights="noisy-student", include_top=False,
			input_shape=input_size)

	headModel = baseModel.output
	headModel = GlobalAveragePooling2D()(headModel)
	headModel = Dense(1024, activation="relu", kernel_initializer="he_uniform", name="fc1")(
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
	model.load_weights(weights)

	model_fc = Model(
		inputs=baseModel.input,
		outputs=model.get_layer("fc1").output
	)

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
	return model_fc


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


def cat_loss(y_true, y_pred):
	return K.categorical_crossentropy(y_true, y_pred)


def joint_loss(y_true, y_pred):
	# mse_loss = K.mean(K.square(y_true - y_pred))
	foc_loss = categorical_focal_loss_fixed(y_true, y_pred, alpha=.25, gamma=2.)
	cat_loss = K.categorical_crossentropy(y_true, y_pred)
	return foc_loss + cat_loss


def scatter(x, labels, subtitle=None):
	# We choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", 200))

	# We create a scatter plot.
	f = plt.figure(figsize=(16, 16))
	ax = plt.subplot(aspect="equal")
	sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[labels.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis("off")
	ax.axis("tight")

	# We add the labels for each digit.
	txts = []
	for i in range(200):
		# Position of each label.
		xtext, ytext = np.median(x[labels == i, :], axis=0)
		txt = ax.text(xtext, ytext, str(i), fontsize=11)
		txt.set_path_effects(
			[PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
		)
		txts.append(txt)

	if subtitle != None:
		plt.suptitle(subtitle)

	plt.savefig(subtitle)


def pairwise_distance(feature, squared=False):
	"""Computes the pairwise distance matrix with numerical stability.

	output[i, j] = || feature[i, :] - feature[j, :] ||_2

	Args:
	  feature: 2-D Tensor of size [number of data, feature dimension].
	  squared: Boolean, whether or not to square the pairwise distances.

	Returns:
	  pairwise_distances: 2-D Tensor of size [number of data, number of data].
	"""
	pairwise_distances_squared = math_ops.add(
		math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
		math_ops.reduce_sum(
			math_ops.square(array_ops.transpose(feature)),
			axis=[0],
			keepdims=True)) - 2.0 * math_ops.matmul(feature,
													array_ops.transpose(feature))

	# Deal with numerical inaccuracies. Set small negatives to zero.
	pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
	# Get the mask where the zero distances are at.
	error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

	# Optionally take the sqrt.
	if squared:
		pairwise_distances = pairwise_distances_squared
	else:
		pairwise_distances = math_ops.sqrt(
			pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

	# Undo conditionally adding 1e-16.
	pairwise_distances = math_ops.multiply(
		pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

	num_data = array_ops.shape(feature)[0]
	# Explicitly set diagonals to zero.
	mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
		array_ops.ones([num_data]))
	pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
	return pairwise_distances

def masked_maximum(data, mask, dim=1):
	"""Computes the axis wise maximum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the maximum.

	Returns:
	  masked_maximums: N-D `Tensor`.
		The maximized dimension is of size 1 after the operation.
	"""
	axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
	masked_maximums = math_ops.reduce_max(
		math_ops.multiply(data - axis_minimums, mask), dim,
		keepdims=True) + axis_minimums
	return masked_maximums

def masked_minimum(data, mask, dim=1):
	"""Computes the axis wise minimum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the minimum.

	Returns:
	  masked_minimums: N-D `Tensor`.
		The minimized dimension is of size 1 after the operation.
	"""
	axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
	masked_minimums = math_ops.reduce_min(
		math_ops.multiply(data - axis_maximums, mask), dim,
		keepdims=True) + axis_maximums
	return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
	del y_true
	margin = 1.
	labels = y_pred[:, :1]

 
	labels = tf.cast(labels, dtype='int32')

	embeddings = y_pred[:, 1:]

	### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
	
	# Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
	# lshape=array_ops.shape(labels)
	# assert lshape.shape == 1
	# labels = array_ops.reshape(labels, [lshape[0], 1])

	# Build pairwise squared distance matrix.
	pdist_matrix = pairwise_distance(embeddings, squared=True)
	# Build pairwise binary adjacency matrix.
	adjacency = math_ops.equal(labels, array_ops.transpose(labels))
	# Invert so we can select negatives only.
	adjacency_not = math_ops.logical_not(adjacency)

	# global batch_size  
	batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

	# Compute the mask.
	pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
	mask = math_ops.logical_and(
		array_ops.tile(adjacency_not, [batch_size, 1]),
		math_ops.greater(
			pdist_matrix_tile, array_ops.reshape(
				array_ops.transpose(pdist_matrix), [-1, 1])))
	mask_final = array_ops.reshape(
		math_ops.greater(
			math_ops.reduce_sum(
				math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
			0.0), [batch_size, batch_size])
	mask_final = array_ops.transpose(mask_final)

	adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
	mask = math_ops.cast(mask, dtype=dtypes.float32)

	# negatives_outside: smallest D_an where D_an > D_ap.
	negatives_outside = array_ops.reshape(
		masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
	negatives_outside = array_ops.transpose(negatives_outside)

	# negatives_inside: largest D_an.
	negatives_inside = array_ops.tile(
		masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
	semi_hard_negatives = array_ops.where(
		mask_final, negatives_outside, negatives_inside)

	loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

	mask_positives = math_ops.cast(
		adjacency, dtype=dtypes.float32) - array_ops.diag(
		array_ops.ones([batch_size]))

	# In lifted-struct, the authors multiply 0.5 for upper triangular
	#   in semihard, they take all positive pairs except the diagonal.
	num_positives = math_ops.reduce_sum(mask_positives)

	semi_hard_triplet_loss_distance = math_ops.truediv(
		math_ops.reduce_sum(
			math_ops.maximum(
				math_ops.multiply(loss_mat, mask_positives), 0.0)),
		num_positives,
		name='triplet_semihard_loss')
	
	### Code from Tensorflow function semi-hard triplet loss ENDS here.
	return semi_hard_triplet_loss_distance

def triplets_loss(y_true, y_pred):
	
#     embeddings = K.cast(embeddings, 'float32')
#     with sess.as_default():
#         print(embeddings.eval())
	
	embeddings = y_pred
	anchor_positive = embeddings[:10]
	negative = embeddings[10:]
#     print(anchor_positive)

	# Compute pairwise distance between all of anchor-positive
	dot_product = K.dot(anchor_positive, K.transpose(anchor_positive))
	square = K.square(anchor_positive)
	a_p_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product  + K.sum(K.transpose(square), axis=0) + 1e-6
	a_p_distance = K.maximum(a_p_distance, 0.0) ## Numerical stability
#     with K.get_session().as_default():
#         print(a_p_distance.eval())
#     print("Pairwise shape: ", a_p_distance)
#     print("Negative shape: ", negative)

	# Compute distance between anchor and negative
	dot_product_2 = K.dot(anchor_positive, K.transpose(negative))
	negative_square = K.square(negative)
	a_n_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product_2  + K.sum(K.transpose(negative_square), axis=0)  + 1e-6
	a_n_distance = K.maximum(a_n_distance, 0.0) ## Numerical stability
	
	hard_negative = K.reshape(K.min(a_n_distance, axis=1), (-1, 1))
	
	distance = (a_p_distance - hard_negative + 0.2)
	loss = K.mean(K.maximum(distance, 0.0))/(2.)

#     with K.get_session().as_default():
#             print(loss.eval())
			
	return loss

def create_base_network(embedding_size):
	"""
	Base network to be shared (eq. to feature extraction).
	"""
	# input_image = Input(shape=image_input_shape)

	main_input = Input(shape=(1024, ))
	x = Dense(512, activation='relu', kernel_initializer='he_uniform')(main_input)
	# x = Dropout(0.1)(x)
	# x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
	x = Dropout(0.1)(x)
	x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
	x = Dropout(0.1)(x)
	y = Dense(embedding_size)(x)
	base_network = Model(main_input, y)

	# plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
	return base_network


if __name__ == "__main__":

	train_data_mean = np.load("../train_data_mean.npy")
	labels = np.load("../train_label.npy")
	new_label = list()

	lb = LabelEncoder()
	onehot = lb.fit_transform(labels)

	im_size = 299
	wt = "../mixup_299_joint.hdf5"
	# Loading model weights
	model = cnn_model(model_name="efn_noisy", img_size=im_size, weights=wt)

	img_path = sys.argv[1]
	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x /= 255
	x -= train_data_mean

	predictions = model.predict(x)
	predictions = np.array(predictions)
	print(predictions.shape)
	
	model_tr = load_model("semiH_ep26_BS256.hdf5",
		custom_objects={'triplet_loss_adapted_from_tf':triplet_loss_adapted_from_tf})
	
	# creating an empty network
	embedding_size = 256
	testing_embeddings = create_base_network(embedding_size=embedding_size)

	# Grabbing the weights from the trained network
	for layer_target, layer_source in zip(testing_embeddings.layers, model_tr.layers[2].layers):
		weights = layer_source.get_weights()
		layer_target.set_weights(weights)
		del weights        

	x_test_after = testing_embeddings.predict(predictions)

	print("Embeddings after training")

	with open('lg_classifier.pkl', 'rb') as predictor:
		voting_loaded = pickle.load(predictor)

	y_pred = voting_loaded.predict(x_test_after)
	# predict = lb.inverse_transform([y_pred])[0]
	# predict = predict.split(".")[1]
	# s = predict.split("_")
	# # print(s)
	# a=list()
	# for i in range(0,len(s)):
	# 	a.append(s[i].title())
	# y = "_".join(a)
	y_predictions = list()
	# y_predictions.append(lb.inverse_transform([y_pred])[0])
	# y_predictions.append(y)
	# print(y_predictions)

	probs = voting_loaded.predict_proba(x_test_after)
	n=5
	top_n = np.argsort(probs)[:,:-n-1:-1]

	for i in range(5):
		predict = lb.inverse_transform(top_n[0])[i]
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
			probab = 100*float(probs[0][top_n[0][i]])
			# f.write("%.3f:%s\n" % (probab, lb.inverse_transform(top_5)[i]))
			f.write("%.3f:%s\n" % (probab, y_predictions[i]))
	# with open( "birds_class.txt", 'w') as f:
	# 	for item in y_predictions:
	# 		f.write("%s\n" % item)


