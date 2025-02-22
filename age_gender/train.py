import pandas as pd
import logging
import argparse
import os
import json
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import TensorBoard

from wide_resnet import WideResNet
from utils import mk_dir, load_data
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

from IPython import embed

# from tensorflow.python import keras

logging.basicConfig(level=logging.DEBUG)
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"


class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1
        elif epoch_idx < self.epochs * 0.5:
            return 0.02
        elif epoch_idx < self.epochs * 0.75:
            return 0.004
        return 0.0008


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--max_age", type=int, default=116,
                        help="maxim range of age in the dataset of training")
    parser.add_argument("--transfer_learning", type=bool, default=False)
    parser.add_argument("--plot_model", type=bool, default=False)


    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=1, #TODO
                        help="number of epochs")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    max_age = args.max_age + 1
    depth = args.depth
    k = args.width
    transfer_learning = args.transfer_learning
    validation_split = args.validation_split
    use_augmentation = args.aug
    initial_weights = '/home/paula/THINKSMARTER_/Model/demographics-model-prediction/pretrained_models/weights.18-4.06.hdf5'
    # weight_file = '/home/paula/THINKSMARTER_/Model/age-gender-estimation-adapted/checkpoints/weights.09-4.32.hdf5'

    _weight_decay = 0.0005
    _use_bias = False
    _weight_init = "he_normal"

    logging.debug("Loading data...")
    image, gender, age, _, image_size, _ = load_data(input_path)
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    y_data_a = np_utils.to_categorical(age, max_age)

    if transfer_learning:

        model = WideResNet(image_size, depth=depth, k=k, units_age=101)()
        model.load_weights(initial_weights)

        inputs = model.input
        flatten = model.layers[-3].output  # capa flatten
        dense1 = Dense(units=2, kernel_initializer=_weight_init, use_bias=_use_bias,
                              kernel_regularizer=l2(_weight_decay), activation="softmax")(flatten)
        dense2 = Dense(units=117, kernel_initializer=_weight_init, use_bias=_use_bias,
                              kernel_regularizer=l2(_weight_decay), activation="softmax")(flatten)
        model = Model(inputs=inputs, outputs=[dense1, dense2])

        # ---------------------------------
        # IDEA: fine tuning (nomes entreno les dos ultimes capes)
        # for layer in model.layers[:-2]:
        #     layer.trainable = False

    else:
        model = WideResNet(image_size, depth=depth, k=k, units_age=max_age)()


    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    if args.plot_model:
        plot_model(model, to_file='experiments_pictures/model_plot.png', show_shapes=True, show_layer_names=True)

    logging.debug("Saving model...")
    mk_dir("models")
    with open(os.path.join("models", "WRN_{}_{}.json".format(depth, k)), "w") as f:
        f.write(model.to_json())

    mk_dir("checkpoints")
    # tensorBoard = TensorBoard(log_dir='events', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs)),
                 ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]


    logging.debug("Running training...")

    data_num = len(X_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    y_data_g = y_data_g[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    if use_augmentation:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255))
        training_generator = MixupGenerator(X_train, [y_train_g, y_train_a], batch_size=batch_size, alpha=0.2,
                                            datagen=datagen)()

        hist = model.fit_generator(generator=training_generator,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(X_test, [y_test_g, y_test_a]),
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)
    else:
        hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
                         validation_data=(X_test, [y_test_g, y_test_a]))

    logging.debug("Saving weights...")
    model.save_weights(os.path.join("models", "WRN_{}_{}.h5".format(depth, k)), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join("models", "history_{}_{}.h5".format(depth, k)), "history")


    with open('history_tmp.txt', 'w') as f:
        for key in hist.history:
            print(key, file=f)
        f.write('\n')
        json.dump(hist.history, f)


if __name__ == '__main__':
    main()
