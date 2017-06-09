import pandas as pd
from os.path import basename, join, isfile, dirname

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical

from evaluation.metrics import plot_ROC_splits, plot_PR_splits
from features.rf_cnn_codes import main as cnn_rf
from metadata import image_to_metadata
from utils.common import get_subdirs, make_sub_dir

CLASSES = {'No': 0, 'Plus': 1}  #, 'Pre-Plus': 2}


def run_cross_val(all_splits, raw_images, out_dir):

    y_pred_all = []
    y_true_all = []

    for i, split_dir in enumerate(sorted(get_subdirs(all_splits))):

        # Place to store the results
        print "Testing on split #{}".format(i)
        results_dir = make_sub_dir(out_dir, basename(split_dir))

        # npy files for ground truth and predictions
        y_true_out = join(results_dir, 'y_true.npy')
        y_pred_out = join(results_dir, 'y_pred.npy')

        if (not isfile(y_true_out)) or (not isfile(y_pred_out)):

            # Define path to model
            cnn_model = join(split_dir, 'Split{}_Model'.format(i), 'Split{}_Model.yaml'.format(i))
            train_data = join(split_dir, 'train.h5')
            test_data = join(split_dir, 'test.h5')

            # Get a list of original image paths that correspond to the test images
            # test_names =  join(split_dir, 'testing.csv')
            # test_mapping = map_test_to_original(test_names, csv_file)

            # Get the test data, and use CNN + RF to predict
            print "Getting RF predictions from CNN features"
            y_true, y_pred, cnn_features = cnn_rf(cnn_model, test_data, raw_images, results_dir, train_data=train_data)
            # roc_auc, fpr, tpr = calculate_roc_auc(y_pred, to_categorical(y_test), cnn_features['classes'], None)

            # Convert ground truth to categorical (one column per class)
            y_true = to_categorical(y_true)  # binarize true labels

            # Serialize predictions
            np.save(y_true_out, y_true)
            np.save(y_pred_out, y_pred)

        else:

            # Load previous results
            print "Loading previous RF predictions"
            y_true = np.load(y_true_out)
            y_pred = np.load(y_pred_out)

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    # ROC curves - all splits
    for class_ in CLASSES.items():
        fig, ax = plt.subplots()
        plot_ROC_splits(y_true_all, y_pred_all, class_)
        plt.savefig(join(out_dir, 'ROC_AUC_{}_AllSplits.png'.format(class_[0])))
        plt.savefig(join(out_dir, 'ROC_AUC_{}_AllSplits.svg'.format(class_[0])))

    # PR curves - all splits
    for class_ in CLASSES.items():
        fig, ax = plt.subplots()
        plot_PR_splits(y_true_all, y_pred_all, class_)
        plt.savefig(join(out_dir, 'PR_AUC_{}_AllSplits.png'.format(class_[0])))
        plt.savefig(join(out_dir, 'PR_AUC_{}_AllSplits.svg'.format(class_[0])))

        # print "~~{}~~".format(class_name)
        # print "Best threshold: {}".format(thresh)
        # print "FPR/TPR: {}/{}".format(fpr, tpr)
        #
        # # Make hard prediction at best threshold
        # y_pred_best = y_pred[:, c] > thresh
        # conf = confusion_matrix(y_true=y_true[:, c], y_pred=y_pred_best)
        # print conf
        # print classification_report(y_true[:, c], y_pred_best)
        # print accuracy_score(y_true[:, c], y_pred_best)

            # If we're predicting on 'Plus', 0 -> No or Pre-Plus, 1 -> Plus
            # If we're predicting on 'No', 0 -> Pre-Plus or Plus, 1 -> No
            # classes = ['No or Pre-Plus', 'Plus'] if class_name == 'Plus' else ['Pre-Plus or Plus', 'No']
            # plot_confusion(conf, classes, join(results_dir, 'confusion_{}'.format(class_name)))
    #
    # plt.title('ROC curve for prediction of "No" and "Plus"')
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([-0.025, 1.025])
    # plt.ylim([-0.025, 1.025])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig(join(out_dir, 'ROC_AUC_Per_Class_Combined.svg'))

    # Save predictions
    # save_predictions(predictions, labels, CLASSES, results_dir)

    # Plot ROC curves for No and Plus classes, combined across all splits
    # fig, ax = plt.subplots()

    # J = {}
    # for class_name, c in CLASSES.items():
    #
    #     if class_name == 'Pre-Plus':
    #         continue
    #
    #     # Concatenate predictions and labels across all splits
    #     pred = np.concatenate([np.asarray(predictions[class_name][x]) for x in range(0, 5)])
    #     labels = np.concatenate([np.asarray(labels[class_name][x]) for x in range(0, 5)])
    #
    #     print pred.shape
    #     print labels.shape
    #
    #     j, fpr, tpr = plot_roc_auc(pred, labels, name=class_name)
    #     print j
    #     print fpr
    #     print tpr
    #
    #     J[class_name] = j
    #
    # print J


def map_test_to_original(test_csv, original_csv, img_path=None):

    original_list = []

    if img_path is None:
        img_path = join(dirname(original_csv), 'Posterior')

    # Load CSV files for test and original data
    test_df = pd.DataFrame.from_csv(test_csv)
    original_df = pd.DataFrame.from_csv(original_csv)

    # For each test image, get its corresponding raw image
    for test_index, test_row in test_df.iterrows():

        original_image = original_df.iloc[test_index]['imageName']  # image name
        class_dir = image_to_metadata(original_image)['class']  # class directory
        original_path = join(img_path, class_dir, original_image)  # full path to image

        original_list.append(original_path)

    return original_list


def save_predictions(predictions, labels, class_dict, out_dir):

    # Save as CSV
    for class_name, c in class_dict.items():

        pred_out = join(out_dir, 'predictions_{}.csv'.format(class_name))
        labels_out = join(out_dir, 'labels_{}.csv'.format(class_name))

        pred_df = pd.DataFrame(predictions[class_name]).T
        labels_df = pd.DataFrame(labels[class_name]).T

        print pred_df
        print labels_df

        pred_df.to_csv(pred_out)
        labels_df.to_csv(labels_out)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-s', '--splits', dest='splits', help='Directory of trained CNNs', required=True)
    parser.add_argument('-r', '--raw-images', dest='raw_images', help='Directory of all raw images', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory for results', required=True)
    args = parser.parse_args()

    run_cross_val(args.splits, args.raw_images, args.out_dir)
