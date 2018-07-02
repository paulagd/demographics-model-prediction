import os
import cv2
import numpy as np
import argparse
from contextlib import contextmanager
import h5py
import scipy.misc as misc
from keras.utils import np_utils
from sklearn.metrics import accuracy_score,mean_squared_error

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from age_gender.utils import load_data
# MODELS
from age_gender.wide_resnet import WideResNet
# from face_network import create_face_network

# DETECTORS
# export PYTHONPATH=$PYTHONPATH/home/paula/THINKSMARTER_/Face_Detector/FaceNet
import extract_faceNet_faces as faceNet

from IPython import embed

# pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
modhash = '89f56a39a78454e96379348bddd78c0d'

# # NOTE: pretrained weights + ethnicity
# ETHNIC = {0: 'Asian', 1: 'Caucasian', 2: "African", 3: "Hispanic"}
# directory_files = ['test_images/','output_cropped_Images/']
# weight_file = "weights.18-4.06.hdf5"
# weights_ethnic_file = "weights_ethnic.hdf5"
# means_ethnic = "means_ethnic.npy"

# NOTE: weights of training the model with an ethnicity dataset
# ETHNIC = {0: 'White', 1: 'Black', 2: "Asian", 3: "Indian", 4: "Others"}
directory_files = ['test_images/age-gender-preds/','output_cropped_Images/'] #TW  = trained weights

weight_file = "pretrained_models/weights.18-4.06.hdf5"
# weight_file = "age_gender/checkpoints/weights.09-4.32.hdf5"

# weights_ethnic_file = "trained_weights/weights_ethnic_v1.hdf5"
# means_ethnic = 'trained_weights/means_ethnic_v1.npy'



def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")

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

    args = parser.parse_args()
    return args


def transform_image_etnicity_to_predict(im):
	means = np.load(means_ethnic)
	im = im - means
	return np.array([im])


def main():
    args = get_args()
    depth = args.depth
    k = args.width
    max_age = args.max_age + 1

    scaled_matrix = np.empty([])

    for files in os.listdir(directory_files[0]):
        if files.endswith('.jpg') or files.endswith('.png'):
            print ('-------Predicting image:  '+files+'-------')
            img = cv2.imread(os.getcwd()+'/'+directory_files[0]+files)
            # for face detection
            output_directory = directory_files[1]+files.split('.')[0]
            if not os.path.exists(output_directory):
                print ("** Creating output_directory in "+output_directory+' ... **')
                os.makedirs(output_directory)
            [pnet, rnet, onet] = faceNet.create_FaceNet_network_Params(args)
            [scaled_matrix , n_faces_detected ,detected_faces_image] = faceNet.faceNet_Detection(img,output_directory, args, pnet, rnet, onet)

            # Load model and weights of AGE-GENDER
            img_size_age_gender = 64
            img_size_ethnicity = 224
            model_age_gender = WideResNet(img_size_age_gender, depth=depth, k=k, units_age=max_age)()
            model_age_gender.load_weights(weight_file)
            #
            # # Load model and weights of ETHNICITY
            # model_ethnicity = create_face_network(nb_class=4, hidden_dim=512, shape=(224, 224, 3))
            # model_ethnicity.load_weights(weights_ethnic_file)

            # Resize the images for each model
            faces = np.empty((len(scaled_matrix), img_size_age_gender, img_size_age_gender, 3))
            # faces1 = np.empty((len(scaled_matrix), img_size_ethnicity, img_size_ethnicity, 3))

            for i in range(len(scaled_matrix)):
                faces[i, :, :, :] = cv2.resize(scaled_matrix[i], (img_size_age_gender, img_size_age_gender))
                # faces1[i, :, :, :] = cv2.resize(scaled_matrix[i], (img_size_ethnicity, img_size_ethnicity))

            # # predict with ethnicity model
            # # TODO : Reduce the number of predictions in a second / minute
            # result_ethn = np.empty((len(faces1),4))
            # for i in range(len(faces1)):
            #     result_ethn[i] = model_ethnicity.predict(transform_image_etnicity_to_predict(faces1[i]))

            #NOTE predict ages and genders of the detected faces
            result_age_gend = model_age_gender.predict(faces)
            predicted_genders = result_age_gend[0]
            ages = np.arange(0, max_age).reshape(max_age, 1)
            predicted_ages = result_age_gend[1].dot(ages).flatten()

            cols = 5
            rows = int(len(scaled_matrix)/cols) + 1

            for i in range(len(scaled_matrix)):
                plt.subplot(rows, cols, i + 1)
                im = faces[i].astype(np.uint8)
                plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                plt.title("{}, {}".format(int(predicted_ages[i]),
                                          "F" if predicted_genders[i][0]>0.5 else "M"))
                plt.axis('off')
                plt.subplots_adjust(hspace=0.6)
                # cv2.imwrite(output_directory+'/'+"{}_{}_".format(int(predicted_ages[i]),
                #                     "F" if predicted_genders[i][0] > 0.5 else "M") +"_id"+str(i)+".jpg", scaled_matrix[i])

            plt.savefig(output_directory+"/result_"+str(int(len(scaled_matrix)))+".png")


                # misc.imsave(output_directory_files[1]+"{}_{}_{}".format(int(predicted_ages[i]),
                #                     "F" if predicted_genders[i][0] > 0.5 else "M",
                #                      ETHNIC[np.argmax(result_ethn[i])]) +"_"+str(i)+".jpg", scaled_matrix[i])


if __name__ == '__main__':
    main()
