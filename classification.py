from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import cv2 as cv
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
import random

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

dataset_path = r'dataset_mixed/'
traget_img_size = (40, 40)
categories = 11

hog_orientations = 8
hog_pixels_per_block = (4, 4)
hog_cells_per_block = (2, 2)
model = LinearSVC()


def load_dataset(img_names, traget_img_size, max_size, istest=False):
    temp, x, y, count = [], [], [], 0

    for img_name in img_names:
        img = io.imread('{}{}/{}'.format(dataset_path,
                                         img_name.split('-')[0], img_name))
        img = cv.resize(img, traget_img_size)

        descriptor = hog(img, orientations=hog_orientations, pixels_per_cell=hog_pixels_per_block,
                         cells_per_block=hog_cells_per_block, feature_vector=True)
      #   if len(descriptor):
      if not istest:
            max_size = max(max_size, descriptor.shape[0])
      temp.append(descriptor)
      y.append(img_name.split('-')[0])
      #   else:
      #       count += 1

    for desc in temp:
        desc = (desc.flatten()).flatten()
        desc = np.pad(desc, (0, max_size-min(max_size, desc.shape[0])))
        x.append(desc)

#     print("{} images has not features in {}".format(
#         count, "test_set" if istest else "training_set"))
    if istest:
        return np.array(x), np.array(y)
    else:
        return np.array(x), np.array(y), max_size


train_x, train_y, test_x, test_y = np.array(
    []), np.array([]), np.array([]), np.array([])

for category in range(categories):
    ls = os.listdir(dataset_path+str(category))
    x, y = train_test_split(ls, test_size=0.2, shuffle=True)
    train_x = np.append(train_x, x)
    test_x = np.append(test_x, y)

train_x, train_y, max_size = load_dataset(train_x, traget_img_size, 0, False)
test_x, test_y = load_dataset(test_x, traget_img_size, max_size, True)

print("training_set shape", train_x.shape, " test_set shape=", test_x.shape)

model = model.fit(train_x, train_y)
print("")
accuracy = model.score(test_x, test_y)
print("accuracy={}%".format(accuracy*100.0))

# tmp = []
# for i in range(5):
#     img = io.imread("{}6-{}.jpg".format(dataset_path,str(i)))
#     descriptor = hog(img,orientations=hog_orientations,pixels_per_cell=hog_pixels_per_block,cells_per_block=hog_cells_per_block,feature_vector=True)
#     print(model.predict(descriptor.reshape(1,-1)))
