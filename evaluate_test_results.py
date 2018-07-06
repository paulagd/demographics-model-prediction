import os
import cv2
import numpy as np
import argparse
from contextlib import contextmanager
import h5py
import scipy.misc as misc
from keras.utils import np_utils
from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error

from age_gender.utils import load_data
# MODELS
from age_gender.wide_resnet import WideResNet
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
# test_folder = '/Users/paulagomezduran/Desktop/MODELS-SERVER/demographics-model-prediction/data/wiki_db.mat'
# test_folder = '/home/paula/THINKSMARTER_/Model/demographics-model-prediction/data/test_set_UTK.mat'


def main(units_age):

	img_size = 64
	depth = 16
	k = 8


	image, gender, age, _, image_size, _ = load_data(test_folder)
	X_data = image
	y_true_gender = gender
	y_true_age = age

	ranges = [0,10,20,30,40,50,60,70,80,90,100,120]

	# y_true_age = age[:40]

	model_age_gender = WideResNet(img_size, depth=depth, k=k, units_age=units_age)()
	model_age_gender.load_weights(weight_file)

	index = np.arange(len(y_true_age))

	rank_index = []
	rank_age = []

	for i in range(len(ranges)-1):
		begining_r = ranges[i]
		end_r = ranges[i+1]-1

		x = y_true_age > begining_r
		y = y_true_age < end_r

		condition = x & y


		if True in condition:
			rank_index.append(index[condition])
			rank_age.append(y_true_age[index[condition]])
		else:
			print("no")
			rank_index.append([])
			rank_age.append([])


	# GENDER
	result_pred = model_age_gender.predict(X_data)
	y_predict_gender = np.argmax(result_pred[0],axis=1)
	gender_acc = accuracy_score(y_true_gender, y_predict_gender)

	# for i in range(len(rank_index))
	count = 0
	for rank in rank_index:
		result_pred = model_age_gender.predict(X_data[rank])
		ages = np.arange(0, units_age).reshape(units_age, 1)
		if result_pred:
			y_predict_age = result_pred[1].dot(ages).flatten()
			age_mSE = mean_squared_error(y_true_age[rank], y_predict_age)
			age_mAE = mean_absolute_error(y_true_age[rank], y_predict_age)
		else:
			age_mSE = []
			age_mAE = []

		begining_r = ranges[count]
		end_r = ranges[count+1]-1
		print ('--------------------------------------------------- ')
		print ('GENDER ACC: ')
		print (gender_acc)
		# print ('AGE mSE of rank '+str(begining_r)+'-'+str(end_r)+': ')
		# print (age_mSE)
		print ('AGE mean Absolute Error of rank '+str(begining_r)+'-'+str(end_r)+': ')
		print (age_mAE)

		count +=1

	#
    # result_pred = model_age_gender.predict(X_data)
    # y_predict_gender = np.argmax(result_pred[0],axis=1)
    # ages = np.arange(0, units_age).reshape(units_age, 1)
    # y_predict_age = result_pred[1].dot(ages).flatten()
	#
    # gender_acc = accuracy_score(y_true_gender, y_predict_gender)
	#
    # age_mSE = mean_squared_error(y_true_age, y_predict_age)
	#
	#
	#
	# # age_mSE_[i] = mean_squared_error(y_true_age[i], y_predict_age[i])
	#
    # age_mAE = mean_absolute_error(y_true_age, y_predict_age)

	# print ('--------------------------------------------------- ')
	# print ('GENDER ACC: ')
	# print (gender_acc)
	# print ('AGE mSE: ')
	# print (age_mSE)
	# print ('AGE age_mAE: ')
	# print (age_mAE)


if __name__ == '__main__':
    main(101)
