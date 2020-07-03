import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from os.path import isfile, join, exists
from os import rename, listdir, rename, makedirs

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../train/")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()


# Create model object in inference mode.# Creat
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird"
]

import glob
import random
# Train Images path
# image_path = "../train_data/"
train_dir = listdir("../train")
images = []

for sub_dir in train_dir:
    images += glob.glob(join("../train", sub_dir, "*.jpg\n"))

print(len(images))
random.Random(22).shuffle(images)
print(images[:15])
# br
counter = 0

for i in images[:]:
    # print(i)
    specie = i.split("/")[-2]
    # print(specie)
    if not exists("../train_crops/" + specie):
        makedirs("../train_crops/" + specie)
    # br
    img = cv2.imread(i)
    result = model.detect([img], verbose=1)
    # print(result)

    res = result[0]

    n_rois = len(res["rois"])
    print(n_rois)

    for j in range(n_rois):
        if res["class_ids"][j] == 15:
            y1, x1, y2, x2 = res["rois"][j]
            crop = img[y1:y2, x1:x2]
            cv2.imwrite("../train_crops/" + specie + "/" + i.split("/")[-1][:-1], crop)
    #         plt.axis("off")
    #         plt.imshow(crop)
        break

    if counter%500==0:
        print("Number of images done", counter)

    counter+=1

    # plt.savefig("crop.png")
    # break


# sdf

# # Load Test images
# for bird_specie in species:
#     specie = join(image_path, bird_specie)

#     files = listdir(specie)
#     # Files sorting
#     files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

#     birds = 1

#     if not exists(join("../mask_rcnn_crops/", bird_specie)):
#     	makedirs(join("../mask_rcnn_crops/", bird_specie))
    	
#     # Detect birds in each class
#     for file in files:

#         img_path = join(specie, file)

#         image = cv2.imread(img_path, 1)

#         result = model.detect([image], verbose=1)

#         res = result[0]

#         number_of_rois = len(res["rois"])

#         imgs = 1

#         for j in range(number_of_rois):

#             # If bird is found append then crop the bird out of that image
#             if res["class_ids"][j] == 15:

#                 y1, x1, y2, x2 = res["rois"][j]

#                 crop = image[y1:y2, x1:x2]

#                 cv2.imwrite(
#                     "../mask_rcnn_crops/"
#                     + bird_specie
#                     + "/"
#                     + str(birds)
#                     + str(imgs)
#                     + ".jpg",
#                     crop,
#                 )

#                 imgs += 1

#         birds += 1
