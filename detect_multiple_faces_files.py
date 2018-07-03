import os
import cv2
import numpy as np
import argparse
# from contextlib import contextmanager
import h5py
import scipy.misc as misc
from keras.utils import np_utils
from IPython import embed

# DETECTORS
# export PYTHONPATH=$PYTHONPATH/home/paula/THINKSMARTER_/Face_Detector/FaceNet
import extract_faceNet_faces as faceNet
import extract_tinyfaces_faces as tinyFaces

# pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
# modhash = '89f56a39a78454e96379348bddd78c0d'

directory_files = ['test_images/','output_cropped_Images/'] #TW  = trained weights
tinyFaces_args = ['weights.pkl','test_images/','predicted_images/', 3, False]

weight_file = "pretrained_models/weights.18-4.06.hdf5"

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--weight_file", type=str, default=None,
    #                     help="path to weight file (e.g. weights.18-4.06.hdf5)")
    # parser.add_argument("--depth", type=int, default=16,
    #                     help="depth of network")
    # parser.add_argument("--width", type=int, default=8,
    #                     help="width of network")
    
    parser.add_argument("--face_detector", type=str, required=True)

    # IDEA: ADD facenet args
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--max_age', type=int,
        help='Max age range of the dataset.', default=100)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)

    # IDEA: ADD tinyfaces args
    parser.add_argument('--prob_thresh', type=float, help='The threshold of detection confidence(default: 0.5).', default=0.5)
    parser.add_argument('--nms_thresh', type=float, help='The overlap threshold of non maximum suppression(default: 0.1).', default=0.1)
    parser.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default=tinyFaces_args[0])
    parser.add_argument('--data_dir', type=str, help='Image data directory.', default=tinyFaces_args[1])
    parser.add_argument('--output_dir', type=str, help='Output directory for images with faces detected.', default=tinyFaces_args[2])
    parser.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=tinyFaces_args[3])
    parser.add_argument('--display', type=bool, help='Display each image on window.', default=tinyFaces_args[4])

    args = parser.parse_args()
    return args


def transform_image_etnicity_to_predict(im):
	means = np.load(means_ethnic)
	im = im - means
	return np.array([im])


def main():

    args = get_args()
    # depth = args.depth
    # k = args.width
    # max_age = args.max_age + 1
    #
    # scaled_matrix = np.empty([])
    #
    [pnet, rnet, onet] = faceNet.create_FaceNet_network_Params(args)

    for files in os.listdir(directory_files[0]):
        if files.endswith('.jpg') or files.endswith('.png'):
            print ('-------Analysing image:  '+files+'-------')
            img = cv2.imread(os.getcwd()+'/'+directory_files[0]+files)
            # for face detection
            output_directory = directory_files[1]
            if not os.path.exists(output_directory):
                print ("** Creating output_directory in "+output_directory+' ... **')
                os.makedirs(output_directory)


            [scaled_matrix , n_faces_detected, detected_faces_image] = faceNet.faceNet_Detection(img,output_directory, args, pnet, rnet, onet)

            print (n_faces_detected)
            #
            # cols = 5
            # rows = int(len(scaled_matrix)/cols) + 1
            #
            # for i in range(len(scaled_matrix)):
            #     plt.subplot(rows, cols, i + 1)
            #     im = faces[i].astype(np.uint8)
            #     plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            #     plt.title("{}, {}".format(int(predicted_ages[i]),
            #                               "F" if predicted_genders[i][0]>0.5 else "M"))
            #     plt.axis('off')
            #     plt.subplots_adjust(hspace=0.6)
            #     cv2.imwrite(output_directory+'/'+"{}_{}_".format(int(predicted_ages[i]),
            #                         "F" if predicted_genders[i][0] > 0.5 else "M") +"_id"+str(i)+".jpg", scaled_matrix[i])
            #
            # plt.savefig(directory_files[1]+"result_"+str(int(len(scaled_matrix)))+".png")
            #
            #
            #     # misc.imsave(output_directory_files[1]+"{}_{}_{}".format(int(predicted_ages[i]),
            #     #                     "F" if predicted_genders[i][0] > 0.5 else "M",
            #     #                      ETHNIC[np.argmax(result_ethn[i])]) +"_"+str(i)+".jpg", scaled_matrix[i])


if __name__ == '__main__':
    main()
