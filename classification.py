from matplotlib.pyplot import show
from commonfunctions import *
from preprocessing import *
from output_handler import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2
import joblib
import os.path
from operator import itemgetter

'''
get note head charachter basedon its position
'''


def getFlatHeadNotePos(staff_lines, note, staff_space, charPos, staff_height, img_o, isBeamOrChord):
    # if charPos[3]-charPos[2] < staff_space:
    #     return [-1]
    img = np.copy(note)
    # show_images([img])
    s_c = np.copy(staff_lines)
    n_c = np.copy(note)
    s_c = s_c > 0
    n_c = n_c > 0
    s_c[charPos[0]:charPos[1], charPos[2]:charPos[3]
        ] = s_c[charPos[0]:charPos[1], charPos[2]:charPos[3]] | n_c
    err = staff_space//8
    # edges = np.copy(note)
    # edges = edges.astype(int)*255
    # show_images([edges])
    # se = sk.morphology.disk(staff_space//16)
    # img = sk.morphology.binary_closing(img, se)
    # img = sp.ndimage.morphology.binary_fill_holes(img)

    # show_images([img])
    # img = sk.morphology.binary_closing(img)
    # img = sk.morphology.binary_dilation(img)
    se = sk.morphology.disk(staff_space//2-2)
    # se[staff_space//4:3*staff_space//4+1,
    #     staff_space//4:3*staff_space//4+1] = 0
    img = sk.morphology.binary_opening(img, se)
    # show_images([img])
    # se = sk.morphology.disk((staff_space//3))
    se = sk.morphology.disk(staff_space//4)
    img = sk.morphology.binary_erosion(img)
    # show_images([img])
    img = sk.morphology.binary_closing(img)
    se = sk.morphology.disk(staff_space//2-3)
    img = sk.morphology.binary_erosion(img, se)
    img = sk.morphology.binary_erosion(img)
    se = sk.morphology.disk(staff_space//8+1)
    img = sk.morphology.binary_dilation(img, se)
    img = sk.morphology.binary_erosion(img)

    bounding_boxes = sk.measure.find_contours(img, 0.8)
    output = []
    #print("no of notes")
    # print(len(bounding_boxes))
    # show_images([img])
    cols = []
    for box in bounding_boxes:
        try:
            [Xmin, Xmax, Ymin, Ymax] = [np.min(box[:, 1]), np.max(
                box[:, 1]), np.min(box[:, 0]), np.max(box[:, 0])]
            ar = (Xmax-Xmin)/(Ymax-Ymin)
            if True:
                r0 = int(Ymin)
                r1 = int(Ymax)
                r0 = max(r0, 0)
                r1 = min(r1, staff_lines.shape[0])
                col = int(Xmin)
                center = (r0+r1)//2
                center2 = staff_lines.shape[1]//2
                # print("center "+str(center))
                # x = np.copy(note)
                # x[center-staff_space//2:center+staff_space//2, :] = False
                # print(staff_lines[center-staff_height:center+staff_height, :])
                # show_images(
                #     [note, x, staff_lines[center-staff_space//2:center+staff_space//2, :]])
                # show_images(
                #     [s_c[center-staff_space//2:center+staff_space//2, :]])
                horz_hist = np.sum(
                    staff_lines[center-staff_height:center+staff_height, :], axis=1)
                # print(horz_hist)
                maximum = np.sum(horz_hist)
                up_arr = runs_of_ones_array(
                    staff_lines[0:center-staff_height//2, center2])
                up = 0
                t = staff_height//2
                for x in up_arr:
                    if x >= t:
                        up += 1
                # print(maximum)
                cols.append(col)
                if maximum < staff_lines.shape[1]:
                    # print("space")
                    if up == 0:
                        x = 0
                        i = center
                        while i < staff_lines.shape[0] and staff_lines[i, center2] == False:
                            x += 1
                            i += 1
                        if x < staff_space:
                            output.append("g2")
                        elif x < 1.5*staff_space:
                            output.append("a2")
                        else:
                            output.append("b2")
                    elif up == 1:
                        output.append("e2")
                    elif up == 2:
                        output.append("c2")
                    elif up == 3:
                        output.append("a1")
                    elif up == 4:
                        output.append("f1")
                    else:
                        x = 0
                        i = center
                        while i >= 0 and staff_lines[i, center2] == False:
                            x += 1
                            i -= 1
                        if abs(x) < staff_space:
                            output.append("d1")
                        else:
                            output.append("c1")
                else:
                    # print("line")
                    if up == 5:
                        x = 0
                        i = center
                        while i >= 0 and staff_lines[i, center2] == False:
                            x += 1
                            i -= 1
                        if abs(x) < staff_space:
                            output.append("d1")
                        else:
                            output.append("c1")
                    elif up == 1:
                        output.append("d2")
                    elif up == 2:
                        output.append("b1")
                    elif up == 3:
                        output.append("g1")
                    elif up == 4:
                        output.append("e1")
                    elif up == 0:
                        x = 0
                        i = center
                        while i < staff_lines.shape[0] and staff_lines[i, center2] == False:
                            x += 1
                            i += 1
                        # print("X "+str(x))
                        if x < staff_space:
                            output.append("f2")
                        elif x < 1.5*staff_space:
                            output.append("a2")
                        else:
                            output.append("b2")
                if isBeamOrChord == 0:
                    output = sorted(output)
                else:
                    newArr = []
                    for i in range(len(cols)):
                        newArr.append([cols[i], output[i]])
                    newArr = sorted(newArr, key=itemgetter(0))
                    output = []
                    for i in range(len(cols)):
                        output.append(newArr[i][1])
        except:
            pass

    output.insert(0, charPos[2])
    return output


'''
check if the note entered in chord or beam
'''


def check_chord_or_beam(img_input, staff_space):
    '''
        **img is assumed to be binarized
        returns:
            0 --> chord
            1 --> beam /16
            2 --> beam /32
        -1 --> neither
    '''

    se = sk.morphology.disk(staff_space//2-1)
    # img = sk.morphology.binary_opening(img_input, se)
    # img = sk.morphology.binary_erosion(img, se)
    # img = sk.morphology.binary_erosion(img)
    # se = sk.morphology.disk(staff_space//4)
    # img = sk.morphology.binary_dilation(img, se)
    img = sk.morphology.binary_opening(img_input, se)
    # show_images([img])
    # se = sk.morphology.disk((staff_space//3))
    se = sk.morphology.disk(staff_space//4)
    img = sk.morphology.binary_erosion(img)
    img = sk.morphology.binary_dilation(img)
    se = sk.morphology.disk(staff_space//2-1)
    img = sk.morphology.binary_erosion(img, se)
    # img = sk.morphology.binary_erosion(img)
    se = sk.morphology.disk(staff_space//8+1)
    img = sk.morphology.binary_dilation(img, se)
    img = sk.morphology.binary_erosion(img)
    # show_images([img])
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


'''
predict the note given
'''


def classfiyimg(img, staff_space):
    out = check_chord_or_beam(img, staff_space)
    if(out == -1):
        features = extract_features(img)
        print(loaded_model.predict([features]))


'''
extract features for the hog classifier
'''


def extract_features(img):
    img = img.astype(int)
    # show_images([img])

    target_img_size = (78, 78)
    img = cv2.resize(img.astype('uint8'), target_img_size)
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 15  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()

    return h.flatten()


'''
load dataset to be trainedon/tested
'''


def load_dataset():
    path_to_dataset = r'dataset_mixed2\dataset_mixed'
    features = []
    labels = []
    img_filenames = os.listdir(path_to_dataset)

    for i, fn in enumerate(img_filenames):
        if fn.split('.')[-1] != 'jpg' and fn.split('.')[-1] != 'bmp' and fn.split('.')[-1] != 'png':
            continue

        label = fn.split('-')[0]
      #  print(label)
        labels.append(label)

        path = os.path.join(path_to_dataset, fn)
        img = cv2.imread(path)
        features.append(extract_features(img))
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(img_filenames)))

    return features, labels


'''
load dataset and run and train dataset
'''


def run_experiment():

    # Load dataset with extracted features
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset()
#    print(features)
#    print(labels)
    print('Finished loading dataset.')

    # Since we don't want to know the performance of our classifier on images it has seen before
    # we are going to withhold some images that we will test the classifier on after training
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed)

    for model_name, model in classifiers.items():
        print('############## Training', model_name, "##############")
        # Train the model only on the training features
        model.fit(train_features, train_labels)
        # save the model to disk
        filename = 'finalized_model.sav'
        joblib.dump(model, filename)

        # Test the model on images it hasn't seen before
        accuracy = model.score(test_features, test_labels)
        print(model_name, 'accuracy:', accuracy*100, '%')
