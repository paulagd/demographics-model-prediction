# import the necessary packages
import cv2 as cv
import numpy as np
import argparse
import os
# import dlib
import time
import keras.backend as K
# import sys
# sys.path.insert(0,'./feelings/')

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from feelings.utils import load_emotion_model, apply_offsets, draw_bounding_box, draw_text, get_color, draw_str

from console_progressbar import ProgressBar
from IPython import embed

import extract_faceNet_faces as faceNet



def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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


if __name__ == '__main__':

    args = get_args()

    im_path = 'test_images/crowd-of-people-walking.png'
    output_directory = 'results/output_feelings'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7
    img_size_emotions = 224

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # detector = dlib.get_frontal_face_detector()
    emotion_model = load_emotion_model('models/model.best.hdf5')

    [pnet, rnet, onet] = faceNet.create_FaceNet_network_Params(args)

    try:
        start = time.time()
        im = cv.imread(im_path)
        if not os.path.exists(output_directory):
            print ("** Creating output_directory in "+output_directory+' ... **')
            os.makedirs(output_directory)
        [scaled_matrix , n_faces_detected, detected_faces] = faceNet.faceNet_Detection(im,output_directory, args, pnet, rnet, onet)

        # Resize the images for each model
        faces = np.empty((len(scaled_matrix), img_size_emotions, img_size_emotions, 3))


        cols = 5
        rows = int(len(scaled_matrix)/cols) + 1

        for i in range(len(scaled_matrix)):
            faces[i, :, :, :] = cv.resize(scaled_matrix[i], (img_size_emotions, img_size_emotions))

        preds = emotion_model.predict(faces)

        for i in range(len(scaled_matrix)):
            class_id = np.argmax(preds[i])
            emotion = class_names[class_id]
            print (emotion)

            plt.subplot(rows, cols, i + 1)
            im = faces[i].astype(np.uint8)
            plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
            plt.title(str(emotion))
            plt.axis('off')
            plt.subplots_adjust(hspace=0.6)

        plt.savefig("results/output_feelings/emotions_picture.png")


        # color = get_color(emotion, prob)
        # # draw_bounding_box(image=im, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
        # draw_text(image=im, color=color, text=emotion)
        # # draw_text(image=im, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=emotion)
        #
        # end = time.time()
        #
        # cv.imwrite("experiments_pictures/test_feelings.jpg", im)


    except KeyboardInterrupt:
        aux =  time.time() - start
        print("Total time: ", aux)
