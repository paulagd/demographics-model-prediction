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

import face_extractors.extract_faceNet_faces as faceNet
import face_extractors.extract_tinyfaces_faces as tinyFaces

# im_path = 'test_images/age-gender-preds/crowd-of-people-walking.png'
im_path = 'test_images/emotions.jpg'
output_directory = 'results/output_feelings'
tinyFaces_args = ['weights.pkl',im_path,output_directory, 3, False]

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # IDEA: ADD facenet args
    # parser.add_argument('--image_size', type=int,
    #     help='Image size (height, width) in pixels.', default=182)
    # parser.add_argument('--max_age', type=int,
    #     help='Max age range of the dataset.', default=100)
    # parser.add_argument('--margin', type=int,
    #     help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    # parser.add_argument('--random_order',
    #     help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    # parser.add_argument('--gpu_memory_fraction', type=float,
    #     help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    # parser.add_argument('--detect_multiple_faces', type=bool,
    #                     help='Detect and align multiple faces per image.', default=True)

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


if __name__ == '__main__':

    args = get_args()

    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7
    img_size_emotions = 224

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # detector = dlib.get_frontal_face_detector()
    emotion_model = load_emotion_model('models/model.best.hdf5')

    # [pnet, rnet, onet] = faceNet.create_FaceNet_network_Params(args)

    try:
        start = time.time()
        im = cv.imread(im_path)
        if not os.path.exists(output_directory):
            print ("** Creating output_directory in "+output_directory+' ... **')
            os.makedirs(output_directory)
        # [scalied_matrix , n_faces_detected, detected_faces] = faceNet.faceNet_Detection(im,output_directory, args, pnet, rnet, onet)

        [scaled_matrix , n_faces_detected, detected_faces_image] = tinyFaces.tinyFaces_Detection(args,im)

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

            # TODO: WRITE EMOTION ON THE IMAGE

            # cv.imwrite('results/output_feelings/'+emotion+'_'+str(i)+'.jpg',detected_faces_image)

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
