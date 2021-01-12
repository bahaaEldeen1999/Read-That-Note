import os

'''
extract useful line from the output of the classification functions lines
'''


def getLineFromArr(line):
    newLine = []
    i = 0
    symbols = ['#', '##', '&', '&&']
    time = ["4", "2"]
    maxTime = 0
    while i < len(line):
        if line[i][1] in time:
            if maxTime >= 2:
                continue
            if len(newLine) == 0:
                newLine.append([line[i][1]])
            else:
                newLine[0][0] += "_"+line[i][1]
            maxTime += 1
        elif line[i][1] in symbols:
            s = line[i+1][1]
            newS = s[0]
            newS += line[i][1]
            newS += s[1:]
            newLine.append([newS])
            i += 1
        elif line[i][1] == '.':
            newLine[len(newLine)-1][len(newLine[len(newLine)-1])-1] += "."
        else:
            if len(line[i]) == 2:
                newLine.append([line[i][1]])
            else:
                if line[i][1][-1] == "4":
                    newL = []
                    for j in range(1, len(line[i])):
                        newL.append(line[i][j])
                    newLine.append(newL)
                else:
                    for j in range(1, len(line[i])):
                        newLine.append([line[i][j]])
        i += 1
    # print(newLine)
    return newLine


'''
write the line into a file in guido format
'''


def outfunction(array, output_file_name):
    file1 = open(output_file_name+".txt", "w")  # write mode
    flagend = 0
    print(len(array))
    if len(array) > 1:
        file1.write("{\n")
        flagend = 1

    indx = 0
    for initarr in array:
        file1.write("[")
        for j in range(len(initarr)):
            if(len(initarr[j]) == 1):
                if(initarr[j][0] == '4_4'):
                    file1.write(' \meter<"4/4">')
                    continue
                elif(initarr[j][0] == '4_2' or initarr[j][0] == '2_4'):
                    file1.write(' \meter<"4/2">')
                    continue
                else:
                    if j != 0:
                        file1.write(" ")
                    file1.write(initarr[j][0])
                    # if(j != len(initarr)-1):
                    #     file1.write(" ")

            else:
                file1.write("{")
                for i in range(len(initarr[j])):
                    file1.write(initarr[j][i])
                    if(i != len(initarr[j])-1):
                        file1.write(",")
                file1.write("} ")
        #print("i "+str(indx)+" len arr "+str(len(array)))
        if indx != len(array)-1 and len(array) > 1:
            file1.write("],\n")
        else:
            file1.write("]\n")
        indx += 1
    if(flagend):
        file1.write("}")
    file1.close()
    return


'''
get filename without its extension
'''


def get_filename_without_extension(file_path):
    file_basename = os.path.basename(file_path)
    filename_without_extension = file_basename.split('.')[0]
    return filename_without_extension
