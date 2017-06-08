import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join, isfile, basename
import cv2
import numpy as np
from utils.common import find_images_by_class
from learning.retina_net import RetiNet

CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def compare_scores(reader_scores, model_scores):

    pass

def get_raw_scores(model_config, test_data):

    # Load model
    model = RetiNet(model_config).model
    imgs_by_class = find_images_by_class(test_data)

    img_arr, img_names, img_classes = [], [], []

    for class_, img_list in imgs_by_class.items():

        for img_path in img_list:

            # Load and prepare image
            img_names.append(basename(img_path))
            img = cv2.imread(img_path)
            img = img.transpose((2, 0, 1))
            img_arr.append(img)

    # Create single array of inputs
    img_arr = np.stack(img_arr, axis=0)

    # Get raw predictions
    y_pred = model.predict_on_batch(img_arr)

    df = pd.DataFrame(data=y_pred, columns=['p(No)', 'p(Plus)', 'p(Pre-Plus)'], index=img_names)
    df['consensus'] = img_classes
    return df


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-m', '--model-config', dest='model_config', help='Model config (.yaml) file', required=True)
    parser.add_argument('-r', '--rf', dest='rf_pkl', help='Random forest (.pkl) file', required=True)
    parser.add_argument('-t', '--test-data', dest='test_data', help='Test data', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)

    args = parser.parse_args()
    raw_scores = get_raw_scores(args.model_config, args.test_data)
    print raw_scores
