from keras.models import Model
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception
import numpy as np
from keras.preprocessing import image
from keras import utils
from PIL import Image
import cv2
import math
import time
import glob
import sys
import pickle
from os import listdir, makedirs
from os.path import join, exists
from sklearn.preprocessing import LabelEncoder
import random
import efficientnet.keras as efn
import argparse
import keras.backend as K


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


def main():

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

    with open('voting_classifier.pkl', 'rb') as predictor:
        voting_loaded = pickle.load(predictor)

    y_pred = voting_loaded.predict(predictions)

    y_predictions = list()

    probs = voting_loaded.predict_proba(predictions)
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


if __name__ == "__main__":
    main()
