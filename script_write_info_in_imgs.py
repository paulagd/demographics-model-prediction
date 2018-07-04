import os
import cv2
import numpy as np
import argparse
import h5py
import scipy.misc as misc

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from IPython import embed


im_path = 'results/scripts/'

output_directory = 'results/output_predicted_faces/'
tinyFaces_args = ['weights.pkl',output_directory, 3, False]

data_frame_file = 'dataFrame.pkl'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    return args


def main():

    for files in os.listdir(im_path):

        data_frame = pd.read_pickle(im_path+files)
        scaled_matrix = data_frame.values
        # shape --> (53, 2)      matrix[0][0].shape = (5,)   matrix[0][1].shape = (182, 182, 3)


        # for i in range(len(scaled_matrix)):
        #     plt.subplot(rows, cols, i + 1)
        #     im = faces[i].astype(np.uint8)
        #     plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        #     plt.title("{}, {}".format(int(predicted_ages[i]),
        #                               "F" if predicted_genders[i][0]>0.5 else "M"))
        #     plt.axis('off')
        #     plt.subplots_adjust(hspace=0.6)
        #     # cv2.imwrite(output_directory+'/'+"{}_{}_".format(int(predicted_ages[i]),
        #     #                     "F" if predicted_genders[i][0] > 0.5 else "M") +"_id"+str(i)+".jpg", scaled_matrix[i])
        #
        # plt.savefig(output_directory+"/result_"+str(int(len(scaled_matrix)))+".png")
        #
        #
        #     # misc.imsave(output_directory_files[1]+"{}_{}_{}".format(int(predicted_ages[i]),
        #     #                     "F" if predicted_genders[i][0] > 0.5 else "M",
        #     #                      ETHNIC[np.argmax(result_ethn[i])]) +"_"+str(i)+".jpg", scaled_matrix[i])


if __name__ == '__main__':
    main()
