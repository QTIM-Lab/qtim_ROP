import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, save, show
from bokeh.palettes import colorblind, d3
from bokeh.models import HoverTool, ColumnDataSource
import pandas as pd
from os.path import join, isfile, basename, splitext
from keras.utils.np_utils import to_categorical
from glob import glob
import cv2
import numpy as np
from scipy.stats import mode
from utils.common import find_images, find_images_by_class
from learning.retina_net import RetiNet, RetinaRF, locate_config
import re
from sklearn.metrics import cohen_kappa_score, roc_curve
from itertools import cycle
from metrics import fpr_and_tpr


CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}
SHEET_CLASSES = {'No': 1, 'Pre-Plus': 2,  'Plus': 3}


def compare_models(models, images, spreadsheet, out_dir):

    # Initialize reader scores
    model_scores = None

    for split_no in range(0, 5):

        split_name = 'split_{}'.format(split_no)
        model_dir = join(models, split_name)

        split_pred = join(out_dir, split_name + '.csv')
        if isfile(split_pred):
            combined = pd.DataFrame.from_csv(split_pred)
        else:
            cnn_predictions = get_model_scores(model_dir, images, prefix=split_name + '_')
            rf_predictions = get_model_scores(model_dir, images, rf_pkl=join(model_dir, 'rf.pkl'), prefix=split_name + '_')

            combined = cnn_predictions.merge(rf_predictions, left_index=True, right_index=True, on='RSD', suffixes=['_CNN', '_RF'])
            combined.to_csv(split_pred)

        if model_scores is None:
            model_scores = combined
        else:
            model_scores = model_scores.merge(combined, left_index=True, right_index=True, on='RSD')

    # Add consensus (ensemble)
    cnn_cols = sorted([col for col in model_scores.columns.values if col.endswith('CNN')], key=lambda x: x.split('_')[1])
    cnn_splits = np.asarray([cnn_cols[i:i + 3] for i in xrange(0, len(cnn_cols), 3)])

    rf_cols = sorted([col for col in model_scores.columns.values if col.endswith('RF')], key=lambda x: x.split('_')[1])
    rf_splits = np.asarray([rf_cols[i:i + 3] for i in xrange(0, len(rf_cols), 3)])

    ensemble_cnn_pred = pd.DataFrame(np.mean(np.dstack([model_scores[split] for split in cnn_splits]), axis=2),
                                     columns=['ensemble_No_CNN', 'ensemble_Pre-Plus_CNN', 'ensemble_Plus_CNN'],
                                     index=model_scores.index)
    ensemble_rf_pred = pd.DataFrame(np.mean(np.dstack([model_scores[split] for split in rf_splits]), axis=2),
                                    columns=['ensemble_No_RF', 'ensemble_Pre-Plus_RF', 'ensemble_Plus_RF'],
                                    index=model_scores.index)

    model_scores = pd.concat([model_scores, ensemble_cnn_pred, ensemble_rf_pred], axis=1)

    # Get reader scores
    reader_scores, origin_names = get_reader_scores(spreadsheet)
    img_names = sorted([basename(x) for x in find_images(join(images, '*'))])
    _, matched = match_names(origin_names, img_names)

    # Move RSD to first position
    cols = list(model_scores.columns)
    cols.insert(0, cols.pop(cols.index('RSD')))
    model_scores = model_scores.ix[:, cols]
    model_scores = model_scores.replace(to_replace={'RSD': SHEET_CLASSES})

    # Re-index model scores using matched names
    model_scores = model_scores.reindex(matched)  # same order as spreadsheet
    reader_scores.insert(1, 'Matched', matched)
    reader_scores = reader_scores.set_index(['Matched'])

    # Save
    merged_scores = pd.concat([reader_scores, model_scores], axis=1)
    merged_scores.to_csv(join(out_dir, 'all_model_scores.csv'))

    calculate_interrater_ROC(reader_scores, model_scores, out_dir)


def calculate_interrater_ROC(reader_scores, model_scores, out_dir):

    # Ground truth
    rsd_labels = to_categorical(model_scores['RSD'].values - 1)
    model_colors = cycle(d3['Category20'][12])

    # Model splits
    model_cols = model_scores.columns.values[1:]
    model_splits = np.asarray([model_cols[i:i + 3] for i in xrange(0, len(model_cols), 3)])

    for class_index, class_name in {0: 'No', 2: 'Plus'}.items():

        # fig, ax = plt.subplots()
        hover = HoverTool(tooltips=[("Reader", "@desc")], names=['readers'])
        fig = figure(title="ROC for '{}'".format(class_name),
                     x_axis_label='False Positive Rate',
                     y_axis_label='True Positive Rate',
                     tools="pan,wheel_zoom,box_zoom,reset,resize,previewsave",
                     plot_width=800, plot_height=600)
        fig.add_tools(hover)

        # Add individual curves for models
        model_names = model_splits[:, class_index]

        for model in model_names:  # for each model

            model_labels = model_scores[model].values
            fpr, tpr, thresh = roc_curve(rsd_labels[:, class_index], model_labels)
            # ax.plot(fpr, tpr, label=model, linestyle=next(model_line_styles))
            fig.line(fpr, tpr, legend=model, line_color=next(model_colors))

        # Get ROC points for each reader
        x, y, desc = [], [], []
        for reader in reader_scores.columns.values[2:]:

            reader_labels = to_categorical(reader_scores[reader].values - 1)
            fpr, tpr = fpr_and_tpr(rsd_labels[:, class_index], reader_labels[:, class_index])
            x.append(fpr)
            y.append(tpr)
            desc.append(reader)
            # ax.scatter(fpr, tpr, marker=marker, color='k', label=reader)

        # Plot reader points
        source = ColumnDataSource(data=dict(
            x=x, y=y, desc=desc
        ))

        fig.diamond('x', 'y', size=20, name='readers', color='black', source=source)

        fig.legend.click_policy = "hide"
        save(fig, filename=join(out_dir, 'bokeh_{}.html'.format(class_name)))

        # plt.title('Receiver operating characteristic for "{}"'.format(class_name))
        # plt.legend(loc='lower right')
        # # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([-0.025, .525])
        # plt.ylim([0.325, 1.025])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.savefig(join(out_dir, 'roc_{}.png'.format(class_name)))


def get_reader_scores(spreadsheet):

    reader_scores = pd.read_excel(spreadsheet, "Sheet1")
    origin_names = reader_scores['Origin'].dropna().values
    keep_cols = reader_scores.columns[:10]
    reader_scores = reader_scores[keep_cols].dropna()

    return reader_scores, origin_names


def compare_scores(spreadsheet, model_scores, img_dir, out_dir):

    # # Open spreadsheet, keeping only the desired columns
    reader_scores, origin_names = get_reader_scores(spreadsheet)

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


def get_model_scores(model_dir, test_data, rf_pkl=None, prefix=''):

    # Load model

    if rf_pkl is not None:
        model_config, rf_pkl = locate_config(model_dir)
        model = RetinaRF(model_config, rf_pkl=rf_pkl)
    else:
        model_config = glob(join(model_dir, '*.yaml'))[0]
        model = RetiNet(model_config)
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

    df = pd.DataFrame(data=y_pred, columns=[prefix + 'No', prefix + 'Plus', prefix + 'Pre-Plus'], index=img_names)
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

    # score_csv = join(args.out_dir, 'raw_scores.csv')
    # if isfile(score_csv):
    #     raw_scores = pd.DataFrame.from_csv(score_csv)
    # else:
    #     raw_scores = get_raw_scores(args.model_dir, args.test_data)
    #     raw_scores.to_csv(join(args.out_dir, 'raw_scores.csv'))

    # compare_scores(args.spreadsheet, raw_scores, args.test_data, args.out_dir)
    compare_models(args.model_dir, args.test_data, args.spreadsheet, args.out_dir)
