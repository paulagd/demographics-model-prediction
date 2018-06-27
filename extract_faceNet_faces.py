# NOTE:RUN this before using this script:
# export PYTHONPATH=$PYTHONPATH/Users/paulagomezduran/Desktop/THINKSMARTER/Face_Detector/FaceNet
# export PYTHONPATH=$PYTHONPATH/home/paula/THINKSMARTER_/face-detectors/FaceNet

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from scipy import misc
# import scipy
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
import cv2
import util

from IPython import embed

# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt

from time import sleep
from scipy.special import expit
# from PIL import Image

def overlay_bounding_boxes(raw_img, refined_bboxes,lw=3):
  """Overlay bounding boxes of face on images.
    Args:
      raw_img:
        A target image.
      refined_bboxes:
        Bounding boxes of detected faces.
      lw:
        Line width of bounding boxes. If zero specified,
        this is determined based on confidence of each detection.
    Returns:
      None.
  """

  # Overlay bounding boxes on an image with the color based on the confidence.
  for r in refined_bboxes:
    _score = expit(r[4])
    cm_idx = int(np.ceil(_score * 255))
    rect_color = [int(np.ceil(x * 255)) for x in util.cm_data[cm_idx]]  # parula
    _lw = lw
    if lw == 0:  # line width of each bounding box is adaptively determined.
      bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
      _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
      _lw = int(np.ceil(_lw * _score))

    _r = [int(x) for x in r[:4]]
    cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), rect_color, _lw)

  raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

def full_img_with_boxes(full_img,boxes,filename,output_dir, counter):
    raw_img = full_img.copy()
    overlay_bounding_boxes(raw_img,boxes)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (1100,700)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 4

    cv2.putText(raw_img, "Counter:"+ str(counter),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

    # cv2.imshow('im',raw_img )
    # cv2.imwrite(os.path.join(output_dir, filename +'_detected.jpg' ),raw_img)
    return raw_img

def create_FaceNet_network_Params(args):
    print('--- Creating FACENET networks and loading parameters ---')
    # os.system('unset PYTHONPATH')
    # os.system('export PYTHONPATH=$PYTHONPATH/Users/paulagomezduran/Desktop/THINKSMARTER/Face_Detector/FaceNet')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
        with tf.Session(config=sess_config) as sess:
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return [pnet, rnet, onet]

def faceNet_Detection(img, output_dir, args, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    nrof_successfully_aligned = 0
    scaled_matrix = np.array([])
    detected_faces = np.array([])


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = 'test_frame'
    output_filename = os.path.join(output_dir, filename+'.png')

    if not os.path.exists(output_filename) and img is not None:
        if img.ndim<2:
            print('Unable to align "%s"' % image_path)
            # continue
        img = img[:,:,0:3]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if args.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))

            scaled_matrix = np.empty((len(det_arr),args.image_size,args.image_size,3))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-args.margin/2, 0)
                bb[1] = np.maximum(det[1]-args.margin/2, 0)
                bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                nrof_successfully_aligned += 1
                filename_base, file_extension = os.path.splitext(output_filename)
                if args.detect_multiple_faces:
                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                else:
                    output_filename_n = "{}{}".format(filename_base, file_extension)
                scaled_matrix[i] = scaled
                # cv2.imwrite(output_filename_n,scaled)
                # text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))

            detected_faces = full_img_with_boxes(img,bounding_boxes,filename,output_dir, nrof_successfully_aligned)
            print ('Number of croped images on the frame : %d' % ( nrof_successfully_aligned))
            nrof_successfully_aligned=0

        else:
            print('Unable to align "%s"' % image_path)

<<<<<<< HEAD
    return [scaled_matrix, nrof_successfully_aligned, detected_faces]
=======
    return [scaled_matrix, nrof_successfully_aligned, bounding_boxes, detected_faces]
>>>>>>> refs/remotes/origin/master
