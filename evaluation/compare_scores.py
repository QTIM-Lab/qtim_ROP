import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join, isfile, basename, splitext
import sys
import cv2
import numpy as np
from utils.common import find_images, find_images_by_class
from learning.retina_net import RetiNet
import re


CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def compare_scores(spreadsheet, img_dir):

    df = pd.read_excel(spreadsheet, "Sheet1")
    sheet_names = df['Origin'].dropna().values
    img_names = sorted([basename(x) for x in find_images(join(img_dir, '*'))])
    _, matched = match_names(sheet_names, img_names)

    # Create new column with matched names
    df.insert(1, 'ImageName', '')
    df['ImageName'][0:len(img_names)] = matched

    print df


def match_names(sheet_names, img_names):

    matched = []

    for idx, str1_full in enumerate(sheet_names):

        str1 = splitext(str(str1_full))[0].lower()
        split_str1 = re.split("\W+|_|-", str1)

        best_match = None
        best_score = 0

        for str2_full in img_names:

            str2 = splitext(str(str2_full))[0].lower()
            split_str2 = re.split("\W+|_|-", str2)

            if str1 == str2:
                best_match = str2_full
                break
            else:

                similarity = 0
                for i in split_str1:
                    for j in split_str2:
                        if i == j:
                            similarity += 1

                if similarity > best_score:
                    best_score = similarity
                    best_match = str2_full

        #  print "{}: {} --> {}".format(idx + 1, str1_full, best_match)
        matched.append([str1_full, best_match])

    matched_arr = np.asarray(matched)
    return matched_arr[: 0], matched_arr[:, 1]


def get_raw_scores(model_config, test_data):

    # Load model
    model = RetiNet(model_config).model
    imgs_by_class = find_images_by_class(test_data)

    img_arr, img_names, img_classes = [], [], []

    for class_, img_list in imgs_by_class.items():

        for img_path in img_list:

            # Load and prepare image
            img_names.append(basename(img_path))
            img_classes.append(class_)
            img = cv2.imread(img_path)
            img = img.transpose((2, 0, 1))
            img_arr.append(img)

    # Create single array of inputs
    img_arr = np.stack(img_arr, axis=0)

    # Get raw predictions
    y_pred = np.around(model.predict_on_batch(img_arr), decimals=3)

    df = pd.DataFrame(data=y_pred, columns=['p(No)', 'p(Plus)', 'p(Pre-Plus)'], index=img_names)
    reordered_cols = [df.columns.values[c] for c in [0, 2, 1]]
    df = df[reordered_cols]

    df['RSD'] = img_classes
    return df


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-m', '--model-config', dest='model_config', help='Model config (.yaml) file', required=True)
    parser.add_argument('-r', '--rf', dest='rf_pkl', help='Random forest (.pkl) file', required=True)
    parser.add_argument('-t', '--test-data', dest='test_data', help='Test data', required=True)
    parser.add_argument('-s', '--spreadsheet', dest='spreadsheet', help='Spreadsheet of reader scores', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)

    args = parser.parse_args()
    #raw_scores = get_raw_scores(args.model_config, args.test_data)
    #raw_scores.to_csv(join(args.out_dir, 'raw_scores.csv'))

    compare_scores(args.spreadsheet, args.test_data)
