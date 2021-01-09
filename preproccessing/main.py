from classification import *
from preprocessing import *
import sys


def mainPipeLine(img_original):
    img = np.copy(img_original)
    return img


if __name__ == "__main__":
    input_folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    try:
        os.mkdir(output_folder_path)
    except:
        pass
    for filename in os.listdir(input_folder_path):
        img = sk.io.imread(os.path.join(
            input_folder_path, filename), as_gray=True)
