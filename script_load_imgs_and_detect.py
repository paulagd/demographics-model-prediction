import os
import cv2
import numpy as np
import argparse
import h5py
from tqdm import tqdm
from pandas import DataFrame

# DETECTORS
import face_extractors.extract_tinyfaces_faces as tinyFaces
from IPython import embed

im_path = 'test_images/age-gender-preds/'
output_directory = 'results/scripts/'
tinyFaces_args = ['weights.pkl',output_directory, 3, False]

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # IDEA: ADD tinyfaces args
    parser.add_argument('--prob_thresh', type=float, help='The threshold of detection confidence(default: 0.5).', default=0.5)
    parser.add_argument('--nms_thresh', type=float, help='The overlap threshold of non maximum suppression(default: 0.1).', default=0.1)
    parser.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default=tinyFaces_args[0])
    # parser.add_argument('--data_dir', type=str, help='Image data directory.', default=tinyFaces_args[1])
    parser.add_argument('--output_dir', type=str, help='Output directory for images with faces detected.', default=tinyFaces_args[1])
    parser.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=tinyFaces_args[2])
    parser.add_argument('--display', type=bool, help='Display each image on window.', default=tinyFaces_args[3])

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    scaled_matrix = np.empty([])

    for files in tqdm(os.listdir(im_path)):
        if files.endswith('.jpg') or files.endswith('.png'):
            print ('-------Detecting image:  '+files+' ... -------')
            img = cv2.imread(os.getcwd()+'/'+im_path+files)

            if not os.path.exists(output_directory):
                print ("** Creating output_directory in "+output_directory+' ... **')
                os.makedirs(output_directory)

            [scaled_matrix , bboxes, detected_faces_image] = tinyFaces.tinyFaces_Detection(args,img)


            # TESTING:
            index = []
            rows = []

            for i in range(len(scaled_matrix)):
                rows.append({'imgs': scaled_matrix[i] , 'bboxes': bboxes[i]})
                index.append(i)

            data_frame = DataFrame(rows, index=index)
            data_frame.to_pickle(output_directory+'dataFrame_'+files.split('.')[0]+'.pkl')


            # cv2.imwrite(output_directory+'CROPPED_FACE.jpg', scaled_matrix[0])
            # cv2.imwrite(output_directory+'FULL_IMG_BB_DETECTION.jpg', detected_faces_image)



if __name__ == '__main__':
    main()
