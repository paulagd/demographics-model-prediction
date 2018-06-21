# import the necessary packages
import cv2 as cv
import numpy as np
import argparse
# import dlib
import time
import keras.backend as K
from utils_em import load_emotion_model
from utils_em import apply_offsets
from utils_em import draw_bounding_box
from utils_em import draw_text
from utils_em import get_color
from utils_em import draw_str
from console_progressbar import ProgressBar
from IPython import embed


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


if __name__ == '__main__':

    im_path = 'test_images/neutral.jpg'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # detector = dlib.get_frontal_face_detector()
    emotion_model = load_emotion_model('models/model.best.hdf5')

    try:
        start = time.time()
        im = cv.imread(im_path)
        # (x, y, w, h) = rect_to_bb(im)
        # x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
        # gray_face = gray[y1:y2, x1:x2]
        # TODO: pasarli la cara en coordenades amb algun detector
        gray_face = cv.resize(im, (img_height, img_width))
        # gray_face = cv.resize(gray_face, (img_height, img_width))
        gray_face = np.expand_dims(gray_face, 0)

        preds = emotion_model.predict(gray_face)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        emotion = class_names[class_id]
        #
        # print(emotion)

        color = get_color(emotion, prob)
        # draw_bounding_box(image=im, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
        draw_text(image=im, color=color, text=emotion)
        # draw_text(image=im, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=emotion)

        end = time.time()

        cv.imwrite("test_feelings.jpg", im)


    except KeyboardInterrupt:
        aux =  time.time() - start
        print("Total time: ", aux)
