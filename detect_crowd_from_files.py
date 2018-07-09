import os
import cv2
import numpy as np
import argparse
# from contextlib import contextmanager
import h5py
import scipy.misc as misc
from keras.utils import np_utils
from IPython import embed

from crowd_counting.crowd_counting import CrowdCounting

directory_files = ['test_images/crowds/','results/output_crowd_images/'] #TW  = trained weights

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    return args

def write_info_to_img(full_img, counter):

    raw_img = full_img.copy()


    toWrite = "Estimated counter:"+ str(counter)
    bottomLeftCornerOfText = (full_img.shape[1]-300,full_img.shape[0]- full_img.shape[0] + 25)


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.8
    lineType               = 2

    if np.mean(full_img[full_img.shape[0]-50,full_img.shape[1]-200,:]) > 180:
        fontColor = (0,0,0)
    else:
        fontColor = (255,255,255)

    cv2.putText(raw_img, toWrite,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

    return raw_img


def main():

    args = get_args()

    for files in os.listdir(directory_files[0]):
        if files.endswith('.jpg') or files.endswith('.png'):
            print ('-------Analysing image:  '+files+'-------')
            img = cv2.imread(os.getcwd()+'/'+directory_files[0]+files)
            # for face detection
            output_directory = directory_files[1]
            if not os.path.exists(output_directory):
                print ("** Creating output_directory in "+output_directory+' ... **')
                os.makedirs(output_directory)

            cc = CrowdCounting()
            number_person = cc.run(img)

            img_written = write_info_to_img(img, number_person[0])

            print ('The estimated number of people is ')
            print(number_person[0])
            cv2.imwrite(output_directory+files,img_written)


if __name__ == '__main__':
    main()
