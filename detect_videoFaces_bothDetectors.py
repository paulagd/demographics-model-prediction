import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
# import util
import time
import sys
import scipy.misc as misc
from scipy.special import expit

from IPython import embed

# import face_extractors.extract_faceNet_faces as faceNet
import face_extractors.extract_tinyfaces_faces as tinyFaces


# pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
modhash = '89f56a39a78454e96379348bddd78c0d'

# # NOTE: pretrained weights + ethnicity
# ETHNIC = {0: 'Asian', 1: 'Caucasian', 2: "African", 3: "Hispanic"}
# directory_files = ['test_images/','output_cropped_Images/']
# weight_file = "weights.18-4.06.hdf5"

# NOTE: weights of training the model with an ethnicity dataset
# ETHNIC = {0: 'White', 1: 'Black', 2: "Asian", 3: "Indian", 4: "Others"}

# MODEL PARAMS
# weight_file = "weights.18-4.06.hdf5"

# VIDEO allocation
# video_directory = '/home/paula/THINKSMARTER_/10km_de_course.wmv.mp4'
video_directory = '/home/paula/THINKSMARTER_/Model/ExtendedTinyFaces/test.avi'
# video_directory = '/Users/paulagomezduran/Desktop/10km_de_course.wmv.mp4'
output_directory = 'results/output_video_frames/'

# weight_file_path / data_dir / output_dir / line_width / display
tinyFaces_args = ['weights.pkl','test_images/','results/predicted_images/', 3, False]


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--face_detector", type=str, required=True)

    # IDEA: ADD MODELS args

    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")

    # IDEA: ADD facenet args
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
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

def main():

    start = time.time()
    args = get_args()
    depth = args.depth
    k = args.width

    capture = cv2.VideoCapture(video_directory)

    # # NOTE: PYTHON 3
    # fourcc = cv2.VideoWriter.fourcc('M','J','P','G')
    # # In this way it always works, because your get the right "size"
    # size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # NOTE: PYTHON 2.7
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, size)
    # out1 = cv2.VideoWriter('output_facenet.avi',fourcc, 20.0, size)
    # out2 = cv2.VideoWriter('output_tinyfaces.avi',fourcc, 20.0, size)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.face_detector == 'facenet':
        out1 = cv2.VideoWriter(output_directory+'output_facenet.avi',fourcc, 20.0, size)
    elif args.face_detector == 'tinyfaces':
        out2 = cv2.VideoWriter(os.path.join(output_directory,'output_tinyfaces_TEST.avi'),fourcc, 20.0, size)

    count = 0
    maxDetectedFaces = []

    try:
        while (capture.isOpened()):
            success,frame = capture.read()
            # print('Read a new frame: ', frame.shape) # --> (720, 1280, 3)

            if args.face_detector == 'facenet':

                # out_fn = cv2.VideoWriter('output_facenet.avi',fourcc, 20.0, size)
                [pnet, rnet, onet] = faceNet.create_FaceNet_network_Params(args)
                [scaled_matrix , n_faces_detected, detected_faces_image] = faceNet.faceNet_Detection(frame,output_directory, args, pnet, rnet, onet)
                # print ('scaled_matrix.shape',scaled_matrix.shape)

                # if detected_faces_image.ndim == 1:
                #     print("------SOMETHING IS WRONG------")
                #     sys.exit(0)
                # else:
                    # IDEA: LOAD MODELS -- SLOW
                    # NOTE[faces, faces1] = loadModels(scaled_matrix, depth, k)
                    # FACES  SHAPEEEE.------ (11, 64, 64, 3)
                    # FACES 11111 SHAPEEEE.------ (11, 224, 224, 3)
                    # cv2.imshow('FRAME',detected_faces_image)
                    # IDEA: save all frames
                    # name = "out_video/frame_"+str(count)+".jpg"
                    # print (name)
                    # cv2.imwrite(name,detected_faces_image)
                out1.write(detected_faces_image)
                count +=1

            elif args.face_detector == 'tinyfaces':
                # out_tf = cv2.VideoWriter('output_tinyfaces.avi',fourcc, 20.0, size)
                try:
                    [scaled_matrix , n_faces_detected, detected_faces_image] = tinyFaces.tinyFaces_Detection(args,frame)
                    # TODO: develop a scaled_matrix


                    # if detected_faces_image.ndim == 1:
                    #     print("------SOMETHING IS WRONG------")
                    # else:
                    out2.write(detected_faces_image)
                    maxDetectedFaces.append(len(n_faces_detected))
                    cv2.imwrite(output_directory+'output_video_tinyFaces/frame_%05d.png' % count, detected_faces_image)
                    count +=1

                except:
                    print ("This is an error message in frame " + str(count))
                    break
            else:
                print ('A face detector is required as an argument. The options are: facenet or tinyfaces.')


        text_file = open("MAX_COUNTER.txt", "w")
        text_file.write("MAXIM COUNTER: %s" % np.max(maxDetectedFaces))
        text_file.close()
        print("Total time: ", time.time() - start)
        # Release everything if job is finished
        capture.release()
        # out1.release()
        # out2.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        aux =  time.time() - start
        capture.release()
        out1.release()
        out2.release()
        print("Total time: ", aux)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
