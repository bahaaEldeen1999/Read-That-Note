from matplotlib.pyplot import show
from classification import *
from preprocessing import *
import sys
import os
from operator import itemgetter


def mainPipeLine(img_original):
    img = np.copy(img_original)
    # deskew the image
    img = deskew(img)
    # convert imagr to unsigned int 8 byte
    img = convertImgToUINT8(img)
    # convert image to binay
    img = binarize(img)
    # get copy of priginal image
    img_o = np.copy(img)
    # convert from True False to 0-1
    img = np.array(img)
    # Get Staff-height, Staffs-Space
    staff_height, staff_space = verticalRunLength(img)
    # get threshold based on staff height and space to remove lines
    T_LEN = staff_height
    # extract filled circle to add it to removed lines
    img_filled_notes = extractCircleNotes(img, staff_height)
    img_filled_notes = img_filled_notes > 0
    # remove lines from image
    removed_staff = extractMusicalNotes(img, T_LEN)
    removed_staff = removed_staff > 0
    removed_staff = removed_staff | img_filled_notes

    # extract staff lines from image
    T_LEN = min(2*staff_height, staff_height+staff_space)
    staffLines = removeMusicalNotes(img, T_LEN)
    # get fixed staff lines
    se = np.ones((1, 55))
    fixed_staff_lines = restoreStaffLines(staffLines, T_LEN, img)
    fixed_staff_lines = sk.morphology.binary_opening(fixed_staff_lines, se)
    fixed_staff_lines = fixStaffLines(
        fixed_staff_lines, staff_height, staff_space, img_o)
    fixed_staff_lines = sk.morphology.binary_opening(fixed_staff_lines, se)
    # get staff lines from image
    lines = classicLineSegmentation(img, staff_space)
    for line in lines:
        lineOut = []
        # get notes bounds from the image without the staff lines
        #show_images([removed_staff[line[0]:line[1], line[2]:line[3]]])
        bounds = char_seg(removed_staff[line[0]:line[1], line[2]:line[3]])
        for char in bounds:
            try:
                # print(char)
                char = [0, removed_staff[line[0]:line[1], line[2]
                    :line[3]].shape[0], int(char[2]), int(char[3])]
                out = getFlatHeadNotePos(fixed_staff_lines[line[0]: line[1], line[2]: line[3]], removed_staff[line[0]: line[1],
                                                                                                              line[2]: line[3]][char[0]: char[1], char[2]: char[3]], staff_space, char, staff_height)
                lineOut.append(
                    out)
            except Exception as e:
                print(e)
                pass
        print("--------Line----------")
        lineOut = sorted(lineOut, key=itemgetter(0))
        print(lineOut)
    return img


if __name__ == "__main__":
    input_folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    try:
        os.mkdir(output_folder_path)
    except:
        pass
    for filename in os.listdir(input_folder_path):
        print(filename)
        img = sk.io.imread(os.path.join(
            input_folder_path, filename), as_gray=True)
        mainPipeLine(img)
