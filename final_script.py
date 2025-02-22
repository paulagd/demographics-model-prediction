import os
import cv2
import numpy as np
import argparse
import h5py
from tqdm import tqdm
from pandas import DataFrame, read_pickle
from scipy.special import expit
from face_extractors.util import cm_data
import pickle

# DETECTORS
import face_extractors.extract_tinyfaces_faces as tinyFaces

# MODELS
from age_gender.wide_resnet import WideResNet
from age_gender.face_network import create_face_network
from feelings.utils import load_emotion_model


from IPython import embed

def get_args(tinyFaces_args):
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # IDEA: ADD tinyfaces args
    parser.add_argument('--prob_thresh', type=float, help='The threshold of detection confidence(default: 0.5).', default=0.5)
    parser.add_argument('--nms_thresh', type=float, help='The overlap threshold of non maximum suppression(default: 0.1).', default=0.1)
    parser.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default=tinyFaces_args[0])
    # parser.add_argument('--data_dir', type=str, help='Image data directory.', default=tinyFaces_args[1])
    # parser.add_argument('--output_dir', type=str, help='Output directory for images with faces detected.', default=tinyFaces_args[1])
    parser.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=tinyFaces_args[1])
    parser.add_argument('--display', type=bool, help='Display each image on window.', default=tinyFaces_args[2])

    args = parser.parse_args()
    return args

def load_and_detect_images(im_path, tinyFaces_args, output_directory):

    args = get_args(tinyFaces_args)
    scaled_matrix = np.empty([])

    results = []

    print ('-------Detecting images ... -------')

    for files in tqdm(os.listdir(im_path)):
        if files.endswith('.jpg') or files.endswith('.png'):

            if os.path.isfile(output_directory+'data/dataFrame_'+files.split('.')[0]+'.pkl'):
                print("-------> Loading DataFrame ...")
                data_frame = read_pickle(output_directory+'data/dataFrame_'+files.split('.')[0]+'.pkl')

            else:
                print ('-------> Imgage '+ files)
                img = cv2.imread(os.getcwd()+'/'+im_path+files)

                [scaled_matrix , bboxes, detected_faces_image] = tinyFaces.tinyFaces_Detection(args,img)

                index = []
                rows = []

                for i in range(len(scaled_matrix)):
                    rows.append({'img':img ,'faces': scaled_matrix[i] , 'bboxes': bboxes[i], 'name_img':files.split('.')[0]})
                    index.append(i)

                data_frame = DataFrame(rows, index=index)
                data_frame.to_pickle(output_directory+'data/dataFrame_'+files.split('.')[0]+'.pkl')

            # results.append([data_frame, img])
            results.append(data_frame)
            # data_frame --> shape (53,3)

    return results

def overlay_bounding_boxes(raw_img, refined_bboxes, demographics, lw=3):
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
  count = 0
  for r in refined_bboxes:
    _score = expit(r[4])
    cm_idx = int(np.ceil(_score * 255))
    rect_color = [int(np.ceil(x * 255)) for x in cm_data[cm_idx]]  # parula
    _lw = lw
    if lw == 0:  # line width of each bounding box is adaptively determined.
      bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
      _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
      _lw = int(np.ceil(_lw * _score))

    _r = [int(x) for x in r[:4]]
    x = _r[0]
    y = _r[1]
    cv2.rectangle(raw_img, (x, y), (_r[2], _r[3]), rect_color, _lw)
    # IDEA:cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 00), 2)
    # cv2.putText(raw_img, demographics, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.putText(raw_img, demographics[count], (x+20, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    # cv2.putText(raw_img, demographics[count], (x, y-5), cv2.FONT_HERSHEY_TRIPLEX, 1, rect_color, lineType=cv2.LINE_AA)
    count += 1
    # cv2.imwrite('EXAMPLE.jpg',raw_img)

def write_info_to_img_bboxes(full_img, boxes, info):

    raw_img = full_img.copy()
    overlay_bounding_boxes(raw_img,boxes,info['demographics'])

    # if info['counter'] == 1:
    # # if info['counter'] > 4:
    #     toWrite = "Emotion:"+ str(info['text'])
    #     bottomLeftCornerOfText = (full_img.shape[1]-300,full_img.shape[0]-50)
    #
    # else:
    toWrite = "Counter:"+ str(info['counter'])
    bottomLeftCornerOfText = (full_img.shape[1]-200,full_img.shape[0]-50)



    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    lineType               = 4

    if np.mean(full_img[full_img.shape[0]-50,full_img.shape[1]-200,:]) > 180:
        fontColor = (0,0,0)
    else:
        fontColor = (255,255,255)

    cv2.putText(raw_img, toWrite,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

    # cv2.imshow('im',raw_img )
    # cv2.imwrite(os.path.join(output_dir, filename +'_detected.jpg' ),raw_img)
    return raw_img

def preProcess_Image(scaled_matrix):

    return scaled_matrix

def predict_demographics_Image(scaled_matrix):

    # NOTE: SETUP PARAMETERS
    # FEELINGS
    # class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    class_names = ['A', 'D', 'F', 'H', 'S', 'Surprise', 'N']
    img_size_emotions = 224
    # # ETHNICITY
    # ETHNIC = {0: 'White', 1: 'Black', 2: "Asian", 3: "Indian", 4: "Others"}
    ETHNIC = {0: 'Asian', 1: 'Caucasian', 2: "African", 3: "Hispanic"}
    img_size_ethnicity = 224
    weights_ethnic_file = "pretrained_models/weights_ethnic.hdf5"
    means_ethnic = 'pretrained_models/means_ethnic.npy'
    # AGE-GENDER
    weight_file_age_gender = "pretrained_models/weights.18-4.06.hdf5"
    img_size_age_gender = 64
    depth = 16      # depth of network
    k = 8           # width of network
    max_age = 101   # It can be 117 for the other dataset and weights

    def _transform_image_etnicity_to_predict(im):
        means = np.load(means_ethnic)
        im = im - means
        return np.array([im])

    # NOTE: LOAD MODELS AND WEIGHTS
    emotion_model = load_emotion_model('models/model.best.hdf5')
    model_age_gender = WideResNet(img_size_age_gender, depth=depth, k=k, units_age=max_age)()
    model_age_gender.load_weights(weight_file_age_gender)
    model_ethnicity = create_face_network(nb_class=4, hidden_dim=512, shape=(img_size_ethnicity, img_size_ethnicity, 3))
    model_ethnicity.load_weights(weights_ethnic_file)


    # NOTE: Resize the images for each model
    faces_e = np.empty((len(scaled_matrix), img_size_emotions, img_size_emotions, 3))
    faces_a_g = np.empty((len(scaled_matrix), img_size_age_gender, img_size_age_gender, 3))
    faces_eth = np.empty((len(scaled_matrix), img_size_ethnicity, img_size_ethnicity, 3))
    for i in range(len(scaled_matrix)):
        faces_e[i, :, :, :] = cv2.resize(scaled_matrix[i], (img_size_emotions, img_size_emotions))
        faces_a_g[i, :, :, :] = cv2.resize(scaled_matrix[i], (img_size_age_gender, img_size_age_gender))
        faces_eth[i, :, :, :] = cv2.resize(scaled_matrix[i], (img_size_ethnicity, img_size_ethnicity))


    # NOTE: Do the predictions
    pred_emotions = emotion_model.predict(faces_e)
    result_age_gend = model_age_gender.predict(faces_a_g)
    predicted_genders = result_age_gend[0]
    ages = np.arange(0, max_age).reshape(max_age, 1)
    predicted_ages = result_age_gend[1].dot(ages).flatten()
    result_ethn = np.empty((len(faces_eth),4))
    for i in range(len(faces_eth)):
        result_ethn[i] = model_ethnicity.predict(_transform_image_etnicity_to_predict(faces_eth[i]))


    # NOTE: Build dataframe with info
    rows = []
    index = []

    for i in range(len(scaled_matrix)):

        # demographic_pred = "{}, {}".format(int(predicted_ages[i]),
        #                           "F" if predicted_genders[i][0]>0.5 else "M")
        demographic_pred = "{}, {}, {}".format(int(predicted_ages[i]),
                            "F" if predicted_genders[i][0] > 0.5 else "M",
                             ETHNIC[np.argmax(result_ethn[i])])

        class_id = np.argmax(pred_emotions[i])
        emotion = class_names[class_id]

        rows.append({'emotions': str(emotion) , 'demographics': demographic_pred })
        # rows.append({'emotions': str(emotion)})
        index.append(i)

    info_frame = DataFrame(rows, index=index)

    return [info_frame.demographics, info_frame.emotions]
    # return ["20, F", info_frame.emotions]

def main():

    # NOTE: Define paths and arguments
    im_path = 'test_images/age-gender-preds/'
    output_directory = 'results/scripts/'
    tinyFaces_args = ['weights.pkl', 3, False]


    if not os.path.exists(output_directory):
        print ("** Creating output_directory in "+output_directory+' ... **')
        os.makedirs(output_directory)

    # NOTE: Load images and apply face detectors
    # Returns an array of results which contain [data_frame, detected_faces_image, img] for each img
    results = load_and_detect_images(im_path, tinyFaces_args, output_directory)

    # NOTE: Process images and get predictions or information to write
    # TODO:Replace matrix `data_frame` inside `results[i]` for the processed one
    # scaled_matrix = preProcess_Image(scaled_matrix)

    # NOTE: Write all info in the image

    info = {'counter': None,
            'text': "",
            'demographics': "" }

    i = 0

    # maxDetectedFaces = []
    print('-------- Predicting demographics... --------')
    for dataFrame in tqdm(results):
        # dataFrame --> 'img' ,'faces' 'bboxes'
        name_file_list = output_directory+'data/demographics_'+dataFrame.name_img[0]+'.txt'
        if os.path.isfile(name_file_list):
            with open(name_file_list, "rb") as fp:   # Unpickling
                l = pickle.load(fp)

        else:
            l = predict_demographics_Image(dataFrame.faces)
            with open(name_file_list, "wb") as fp:   #Pickling
                pickle.dump(l, fp)

        # [demographics, emotions]
        demographics = l[0]
        emotions = l[1]

        # data_frame.faces = scaled_matrix
        info['counter'] = len(dataFrame)
        info['text'] = emotions.values[0]
        # info['demographics'] = emotions.values #--> TO SEE EMOTIONS WRITTEN IN BOXES
        info['demographics'] = demographics.values

        # IDEA:Write emotions in demographics
        info['demographics'] = np.column_stack((info['demographics'],emotions.values))
        info['demographics'] = np.array([", ".join(i) for i in info['demographics']])

        imageen = write_info_to_img_bboxes(dataFrame.img[0], dataFrame.bboxes.values, info)
        i +=1
        # maxDetectedFaces.append(len(dataFrame))
        cv2.imwrite(output_directory+'EXAMPLE__'+str(i)+'.jpg',imageen)


    # cv2.imwrite(output_directory+'MAX_COUNTER.txt',np.max(maxDetectedFaces))

if __name__ == '__main__':
    main()
