import os
import cv2
import numpy as np
import argparse
from contextlib import contextmanager
import h5py
import scipy.misc as misc
from utils import load_data
from keras.utils import np_utils

from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error

# MODELS
from wide_resnet import WideResNet
# from face_network import create_face_network

from IPython import embed

# NOTE:
weight_file = "pretrained_models/weights.18-4.06.hdf5"
# weight_file = "checkpoints/weights.02-3.83.hdf5" #FINE TUNING
# weight_file = "checkpoints/weights.09-4.32.hdf5"  #TRANSFER LEARNING

# weights_ethnic_file = "trained_weights/weights_ethnic_v1.hdf5"
# means_ethnic = 'trained_weights/means_ethnic_v1.npy'

# test_folder = '/home/paula/THINKSMARTER_/Model/demographics-model-prediction/data/imdb_db.mat'
test_folder = '/home/paula/THINKSMARTER_/Model/demographics-model-prediction/data/wiki_db.mat'
# test_folder = '/home/paula/THINKSMARTER_/Model/demographics-model-prediction/data/test_set_UTK.mat'


def transform_image_etnicity_to_predict(im):
	means = np.load(means_ethnic)
	im = im - means
	return np.array([im])


def main(units_age):

    img_size = 64
    depth = 16
    k = 8


    image, gender, age, _, image_size, _ = load_data(test_folder)
    X_data = image
    y_true_gender = gender
    y_true_age = age
    model_age_gender = WideResNet(img_size, depth=depth, k=k, units_age=units_age)()
    model_age_gender.load_weights(weight_file)

    result_pred = model_age_gender.predict(X_data)
    y_predict_gender = np.argmax(result_pred[0],axis=1)
    ages = np.arange(0, units_age).reshape(units_age, 1)
    y_predict_age = result_pred[1].dot(ages).flatten()

    gender_acc = accuracy_score(y_true_gender, y_predict_gender)

    age_mSE = mean_squared_error(y_true_age, y_predict_age)
    age_mAE = mean_absolute_error(y_true_age, y_predict_age)

    print ('--------------------------------------------------- ')
    print ('GENDER ACC: ')
    print (gender_acc)
    print ('AGE mSE: ')
    print (age_mSE)
    print ('AGE age_mAE: ')
    print (age_mAE)
	#
	# cols, rows = 2, 1
    # img_num = cols * rows
	#
    # for i in range(img_num):
    #     plt.subplot(rows, cols, i + 1)
    #     plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
    #     plt.title("{}, {}".format(int(predicted_ages[i]),
    #                               "F" if predicted_genders[i][0]>0.5 else "M"))
    #     plt.axis('off')
    # plt.savefig("result.png")


if __name__ == '__main__':
    main(101)
