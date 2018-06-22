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

directory_files = ['test_images/','output_cropped_Images/'] #TW  = trained weights



def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()
    return args



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
            print ('The estimated number of people is ')
            print(number_person[0])


if __name__ == '__main__':
    main()
