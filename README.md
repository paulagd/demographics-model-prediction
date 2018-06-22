# demographics-model-prediction

### Before starting...

There are some folders which should be downloaded in order to get different weights or testing images. The required folders are listed below and they can be found in [here]().

  - data/
  - Datasets/
  - models/
  - pretrained_models/
  - test_images/

### General information

Some scripts are built in python notebook format in order to ease some functionalities.

  1. **split_data.ipynb** --> By providing the path of the desired dataset, it split it into a
  training set (70%) and a test set (30%).

  2. **Dataset_to_db.ipynb** --> By providing the path of the desired dataset (or part of it, as for instance the training set or the test set folders), it creates a .mat database with the desired structure of the model.

### Functionalities of the repository

Different functionalities are implemented in the repository by using different models as well:

  1. **train.py**

  It is stored in the `age_gender` folder and it is used to train a model from scratch, re-train it,
  fine tune it, or compute transfer learning by making the model learn from extended new data.

    -	The weights are initialized from the model trained from scratch with IMDB dataset (https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5)
    but it can be changed.

    -	Check the args before training.

    ```
    python3 train.py --input ../data/imdb_db.mat
    ```

  2. **evaluate_test_results.py**

  It can be used to evaluate the metrics `accuracy` and `mean average error`.

	 - Test on test_set_db.mat or wiki_db.mat amongst others.

    ```
    python3 evaluate_test_results.py
    ```

  3. **predict_multiple_faces_files.py**

  It can be used to predict images with multiple faces on them. It predicts everyone’s
  age and gender (race and feelings will be implemented in the future).

    -	It will generate a .png picture with a resume of the results as well as a folder inside
    `output_cropped_Images` with all the cropped faces individually predicted.

    ```
    python3 predict_multiple_faces_files.py
    ```

  4. **predict_emotion_from_file.py**

  It can be used to predict an emotion from an image file.


  5. **detect_crowd_from_files.py**

  It can be used to estimate crowds on image files.

### Face Detectors

  1. FaceNet

  2. Tiny Faces
