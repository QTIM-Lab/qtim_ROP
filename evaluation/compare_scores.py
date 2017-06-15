import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join, isfile, basename, splitext
import sys
import cv2
import numpy as np
from scipy.stats import mode
from utils.common import find_images, find_images_by_class
from learning.retina_net import RetiNet, RetinaRF, locate_config
import re
from sklearn.metrics import cohen_kappa_score


CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}
SHEET_CLASSES = {'No': 1, 'Pre-Plus': 2,  'Plus': 3}


def compare_scores(spreadsheet, model_scores, img_dir, out_dir):

    # Open spreadsheet, keeping only the desired columns
    reader_scores = pd.read_excel(spreadsheet, "Sheet1")
    origin_names = reader_scores['Origin'].dropna().values
    keep_cols = reader_scores.columns[:10]
    reader_scores = reader_scores[keep_cols].dropna()

    # Calculate their consensus as mode score
    reader_names = reader_scores.columns[2:]
    reader_consensus = mode(reader_scores[reader_names].values, axis=1)[0]

    # Anonymize reader names and add consensus column
    anon_names = {reader: 'Reader #{}'.format(i + 1) for i, reader in enumerate(reader_names)}
    reader_scores.rename(columns=anon_names, inplace=True)
    reader_scores['Consensus'] = reader_consensus

    # Get the raw model scores
    dl_scores = model_scores[['p(No)', 'p(Pre-Plus)', 'p(Plus)']].values
    arg_max = np.argmax(dl_scores, axis=1) + 1  # No: 0, Pre-Plus: 1, Plus: 2
    reader_scores['DL'] = arg_max

    # Create new column with matched names
    img_names = sorted([basename(x) for x in find_images(join(img_dir, '*'))])
    _, matched = match_names(origin_names, img_names)
    model_scores = model_scores.reindex(matched)  # same order as spreadsheet
    reader_scores.insert(1, 'Matched', matched)

    # Add RSD labels
    rsd = [SHEET_CLASSES[x] for x in model_scores['RSD'].values]
    reader_scores['RSD'] = rsd

    # Get reader names
    reader_names = reader_scores.columns.values[3:]
    no_readers = len(reader_names)

    kappa_map = np.zeros(shape=(no_readers, no_readers))

    for i, reader0 in enumerate(reader_names):
        for j, reader1 in enumerate(reader_names):

            r0_scores = reader_scores[reader0].values[:100]
            r1_scores = reader_scores[reader1].values[:100]

            k = cohen_kappa_score(r0_scores, r1_scores, weights='quadratic')
            kappa_map[i, j] = k

    # Plot readers
    # anon_readers = ['Reader #{}'.format(i) for i in range(0, no_readers)]
    df_kappa = pd.DataFrame(kappa_map, columns=reader_names, index=reader_names)
    h = sns.heatmap(df_kappa, annot=True)
    h.set_xticklabels(h.get_xticklabels(), rotation=30)
    plt.gca().xaxis.tick_top()
    sns.plt.savefig(join(out_dir, 'kappa_heatmap.png'))

    reader_scores.to_csv(join(out_dir, 'combined_scores.csv'))


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


def get_raw_scores(model_dir, test_data):

    # Load model
    model_config, rf_pkl = locate_config(model_dir)
    model = RetinaRF(model_config, rf_pkl=rf_pkl)
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
    y_pred = model.predict(img_arr)

    df = pd.DataFrame(data=y_pred, columns=['p(No)', 'p(Plus)', 'p(Pre-Plus)'], index=img_names)
    reordered_cols = [df.columns.values[c] for c in [0, 2, 1]]
    df = df[reordered_cols]

    df['RSD'] = img_classes
    return df


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-m', '--model-dir', dest='model_dir', help='Model directory', required=True)
    parser.add_argument('-t', '--test-data', dest='test_data', help='Test data', required=True)
    parser.add_argument('-s', '--spreadsheet', dest='spreadsheet', help='Spreadsheet of reader scores', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)

    args = parser.parse_args()

    score_csv = join(args.out_dir, 'raw_scores.csv')
    if isfile(score_csv):
        raw_scores = pd.DataFrame.from_csv(score_csv)
    else:
        raw_scores = get_raw_scores(args.model_dir, args.test_data)
        raw_scores.to_csv(join(args.out_dir, 'raw_scores.csv'))

    compare_scores(args.spreadsheet, raw_scores, args.test_data, args.out_dir)
