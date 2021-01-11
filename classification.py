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
import pickle

# global vars
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

dataset_path = r'dataset/'
traget_img_size = (40, 40)
categories = 34

hog_orientations = 8
hog_pixels_per_block = (4, 4)
hog_cells_per_block = (2, 2)
model = None
train_x, train_y, test_x, test_y = np.array(
    []), np.array([]), np.array([]), np.array([])


def check_chord_or_beam(img_input, staff_space):
          '''
        **img is assumed to be binarized
        returns:
            0 --> chord
            1 --> beam /16
            2 --> beam /32
           -1 --> neither
    '''

    se = sk.morphology.disk(staff_space//2)
    img = sk.morphology.binary_opening(img_input, se)
    img = sk.morphology.binary_erosion(img, se)
    img = sk.morphology.binary_erosion(img)
    se = sk.morphology.disk(staff_space//4)
    img = sk.morphology.binary_dilation(img, se)
    bounding_boxes = sk.measure.find_contours(img, 0.8)

    if len(bounding_boxes) < 2:
        return -1

    newImg = img.copy()
    centers, count_disks_spacing = [], 0

    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = [np.min(box[:, 1]), np.max(
            box[:, 1]), np.min(box[:, 0]), np.max(box[:, 0])]
        centers.append([Ymin+Ymin//2, Xmin+Xmin//2])

    for i in range(1, len(centers)):
        if abs(centers[i][1] - centers[i-1][1]) > 2*staff_space:
            count_disks_spacing += 1

    if count_disks_spacing != len(centers)-1:
        return 0

    img = sk.morphology.thin(sk.img_as_bool(img_input))
    h, theta, d = sk.transform.hough_line(img)
    h, theta, d = sk.transform.hough_line_peaks(h, theta, d)
    angels = np.rad2deg(theta)
    number_of_lines = np.sum(np.abs(angels) > 10)

    if number_of_lines < 1 or number_of_lines > 2:
        return -1
    else:
        return number_of_lines



def load_dataset(img_names, traget_img_size, max_size, istest=False):
    temp, x, y, count = [], [], [], 0

    for img_name in img_names:
        img = io.imread('{}{}/{}'.format(dataset_path,
                                         img_name.split('-')[0], img_name))
        img = cv.resize(img, traget_img_size)

        descriptor = hog(img, orientations=hog_orientations, pixels_per_cell=hog_pixels_per_block,
                         cells_per_block=hog_cells_per_block, feature_vector=True)

        if not istest:
            max_size = max(max_size, descriptor.shape[0])
        temp.append(descriptor)
        y.append(img_name.split('-')[0])

    for desc in temp:
        desc = (desc.flatten()).flatten()
        desc = np.pad(desc, (0, max_size-min(max_size, desc.shape[0])))
        x.append(desc)

    if istest:
        return np.array(x), np.array(y)
    else:
        return np.array(x), np.array(y), max_size


if __name__ == "__main__":
    if os.path.exists("classifier.pickle"):
        print("loading the model")
        model = pickle.load(open("classifier.pickle", "rb"))
    else:
        print("training new model")
        model = LinearSVC()
        for category in range(categories):
            ls = os.listdir(dataset_path+str(category))
            x, y = train_test_split(ls, test_size=0.2, shuffle=True)
            train_x = np.append(train_x, x)
            test_x = np.append(test_x, y)

        train_x, train_y, max_size = load_dataset(
            train_x, traget_img_size, 0, False)
        test_x, test_y = load_dataset(test_x, traget_img_size, max_size, True)

        print("training_set shape", train_x.shape,
              " test_set shape", test_x.shape)

        model = model.fit(train_x, train_y)

        # save trained model
        with open("classifier.pickle", "wb") as file:
            pickle.dump(model, file)

    predctions = model.predict(test_x)
    print(predctions)
    # accuracy = model.score(test_x, test_y)
    # print("accuracy={}%".format(accuracy*100.0))
