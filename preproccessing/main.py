from matplotlib.pyplot import show
from classification import *
from preprocessing import *
import sys
import os
from operator import itemgetter


def mainPipeLine(filename, img_original):
    # load model
    if os.path.isfile('finalized_model.sav'):
        ##### the model already exit####
        print("model already exit")
    else:
        ### not exit so train ###
        run_experiment()

    # load the model from disk
    loaded_model = joblib.load('finalized_model.sav')
    # get copy of image
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
    #img = np.array(img)
    # Get Staff-height, Staffs-Space

    #################################
    # work on multiple images       #
    #################################
    patch_height = img.shape[0]//2
    patch_width = img.shape[1]//10
    ph = int(img.shape[0]/patch_height)
    pw = int(img.shape[1]/patch_width)
    removed_staff_c = np.copy(img)
    staff_height_g, staff_space_g = verticalRunLength(img)
    for i in range(ph):
        for j in range(pw):
            try:
                # print("here")
                patch = img[i*patch_height: (i+1)*patch_height,
                            j*patch_width: (j+1)*patch_width]
                staff_height, staff_space = verticalRunLength(patch)
                # get threshold based on staff height and space to remove lines
                T_LEN = staff_height
                # extract filled circle to add it to removed lines
                img_filled_notes = extractCircleNotes(patch, staff_height)
                img_filled_notes = img_filled_notes > 0
                # remove lines from image
                removed_staff = extractMusicalNotes(patch, T_LEN)
                removed_staff = removed_staff > 0
                removed_staff = removed_staff | img_filled_notes
                # io.imsave("test_newOut/patch_"+str(i)+"_"+str(j)+".png", sk.img_as_uint(
                #     removed_staff))
                removed_staff_c[i*patch_height: (i+1)*patch_height,
                                j*patch_width: (j+1)*patch_width] = removed_staff
            except:
                pass
    # show_images([removed_staff_c])
    #removed_staff_c = sk.morphology.binary_opening(removed_staff_c)
    io.imsave("test_newOut/removed_staff"+filename+".png", sk.img_as_uint(
        removed_staff_c))
    print("done "+filename)
    # return
    #########################################
    # extract staff lines from image
    T_LEN = min(2*staff_height_g, staff_height_g+staff_space_g)
    staffLines = removeMusicalNotes(img, T_LEN)
    # get fixed staff lines
    se = np.ones((1, 55))
    fixed_staff_lines = restoreStaffLines(staffLines, T_LEN, img_o)
    fixed_staff_lines = sk.morphology.binary_opening(fixed_staff_lines, se)
    fixed_staff_lines = fixStaffLines(
        fixed_staff_lines, staff_height_g, staff_space_g, img_o)
    fixed_staff_lines = sk.morphology.binary_opening(fixed_staff_lines, se)
    # get staff lines from image
    lines = classicLineSegmentation(img, staff_space_g)
    i = 0
    try:
        os.mkdir("test_newOut/"+filename+"_img")
    except:
        pass
    linesOut = []
    for line in lines:
        lineOut = []
        # get notes bounds from the image without the staff lines
        #show_images([removed_staff_c[line[0]:line[1], line[2]:line[3]]])
        bounds, only_char_arr = char_seg(
            removed_staff_c[line[0]:line[1], line[2]:line[3]])

        #bounds = sorted(bounds,)
        i += 1
        try:
            os.mkdir("test_newOut/"+filename +
                     "_img/"+filename+"_lines"+str(i))
        except Exception as e:
            pass
        # print("here")
        j = 0
        for char in bounds:
            if char[0] == 99999999:
                continue
            charImg = only_char_arr[j]
            j += 1
            # check not garbage
            summation = np.sum(charImg)
            if summation >= 0.98*charImg.shape[0]*charImg.shape[1]:
                continue
            #i += 1
            try:
                #######
                # classify note
                char = [0, removed_staff_c[line[0]:line[1], line[2]                                           :line[3]].shape[0], int(char[2])-2, int(char[3])+2]
                # a,b,c,d ==> wrong classificion
                # start - # - ##  - & - && - end - timestamp - empty string True => end
                # False => get position
                symbols = ['#', '##', '&', '&&', '', '.']
                open_divider = ["_1", "_2"]
                time_stamp = ['4', '2']

                symbol = loaded_model.predict([extract_features(charImg)])[0]
                # check if cord or beam
                isBeamOrChord = check_chord_or_beam(charImg, staff_space_g)
                print(symbol)
                # start == [
                if symbol in time_stamp:
                    lineOut.append([char[2], symbol])
                elif symbol == '[' and isBeamOrChord == -1:
                    # check if image is rotated
                    pass
                elif symbol in symbols and isBeamOrChord == -1:
                    # add to array of charachters
                    if symbol == '':
                        continue
                    lineOut.append([char[2], symbol])
                    pass
                elif symbol in open_divider and isBeamOrChord == -1:
                    # add a/1 or a/2
                    lineOut.append([char[2], "a1"+symbol.replace("_", "/")])
                    pass
                else:
                    # get manual pos
                    pos = getFlatHeadNotePos(fixed_staff_lines[line[0]: line[1], line[2]: line[3]], removed_staff_c[line[0]
                                             : line[1], line[2]: line[3]][char[0]: char[1], char[2]: char[3]], staff_space_g, char, staff_height_g)
                    if len(pos) == 1:
                        continue
                    # check if pos freater than 2 then check if beam or ched
                    if isBeamOrChord == -1:
                        lineOut.append(
                            [pos[0], pos[1]+symbol.replace("_", "/")])
                    elif isBeamOrChord == 0:
                        # then its a chord
                        charsOut = [char[2]]
                        for k in range(1, len(pos)):
                            charsOut.append(pos[k]+"/4")
                        lineOut.append(charsOut)
                    elif isBeamOrChord == 1:
                        # beam /16
                        charsOut = [char[2]]
                        for k in range(1, len(pos)):
                            charsOut.append(pos[k]+"/8")
                        lineOut.append(charsOut)
                    else:
                        # beam /16
                        charsOut = [char[2]]
                        for k in range(1, len(pos)):
                            charsOut.append(pos[k]+"/16")
                        lineOut.append(charsOut)

                    # print(pos)
                    #print(check_chord_or_beam(charImg, staff_space))
                # show_images([charImg])
                # lineOut = sorted(lineOut, key=itemgetter(0))
                # print(lineOut)
                # getFlatHeadNotePos(fixed_staff_lines[line[0]: line[1], line[2]: line[3]], removed_staff[line[0]: line[1],
                #                                                                                         line[2]: line[3]][char[0]: char[1], char[2]: char[3]], staff_space, char, staff_height)
            except Exception as e:
                print(e)
                pass
            #######
            # try:
            #     # print(char)
            #     # show_images(
            #     #    [fixed_staff_lines[line[0]: line[1], line[2]: line[3]]])
            #     char = [0, removed_staff[line[0]:line[1], line[2]
            #         :line[3]].shape[0], int(char[2])-2, int(char[3])+2]
            #     out = getFlatHeadNotePos(fixed_staff_lines[line[0]: line[1], line[2]: line[3]], removed_staff[line[0]: line[1],
            #                                                                                                   line[2]: line[3]][char[0]: char[1], char[2]: char[3]], staff_space, char, staff_height)
            #     lineOut.append(
            #         out)
            # except Exception as e:
            #     print(e)
            #     pass
        print("--------Line----------")
        lineOut = sorted(lineOut, key=itemgetter(0))
        print(lineOut)
        # lineS = ""
        # for arr in lineOut:
        #     if arr[0] == -1:
        #         continue
        #     a = arr[1:]
        #     a = np.sort(a)
        #     if not len(a):
        #         continue
        #     lineS += "{ "
        #     for x in a:
        #         lineS += x+" "
        #     lineS += "}"
        # print(lineS)
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
        mainPipeLine(filename, img)
