import pandas as pd
from os.path import basename, join, isfile, dirname
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from ..evaluation.metrics import plot_ROC_splits
from ..utils.metadata import image_to_metadata
from ..learning.retina_net import RetiNet, RetinaRF
from ..utils.common import get_subdirs, make_sub_dir

CLASSES = {'Normal (vs. pre-plus or plus)': 0, 'Plus (vs. pre-plus or normal)': 1}  #, 'Pre-Plus': 2}


def run_cross_val(all_splits, raw_images, out_dir, use_rf=False):

    y_pred_all = []
    y_true_all = []

    for i, split_dir in enumerate(sorted(get_subdirs(all_splits))):

        # Place to store the results
        print("Testing on split #{}".format(i))
        results_dir = make_sub_dir(out_dir, basename(split_dir))

        # npy files for ground truth and predictions
        y_true_out = join(results_dir, 'y_true.npy')
        y_pred_out = join(results_dir, 'y_pred.npy')

        if (not isfile(y_true_out)) or (not isfile(y_pred_out)):

            # Define path to model
            cnn_model = glob(join(split_dir, '*.yaml'))[0]
            #cnn_model = join(split_dir, 'Split{}_Model.yaml'.format(i))
            train_data = join(split_dir, 'train.h5')
            test_data = join(split_dir, 'test.h5')

            # Get model predictions
            if use_rf:
                print("Using CNN + RF for prediction")
                rf_pkl = join(split_dir, 'rf.pkl')
                model = RetinaRF(cnn_model, rf_pkl=rf_pkl)
            else:
                print("Using CNN only")
                model = RetiNet(cnn_model)

            data_dict = model.evaluate(test_data)
            y_true = data_dict['y_true']
            y_pred = data_dict['y_pred']

            # Serialize predictions
            np.save(y_true_out, y_true)
            np.save(y_pred_out, y_pred)

        else:

            # Load previous results
            print("Loading previous predictions")
            y_true = np.load(y_true_out)
            y_pred = np.load(y_pred_out)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    # ROC curves - all splits
    for class_ in list(CLASSES.items()):
        fig, ax = plt.subplots()
        all_aucs = plot_ROC_splits(y_true_all, y_pred_all, class_)
        plt.savefig(join(out_dir, 'ROC_AUC_{}_AllSplits.png'.format(class_[0])))
        plt.savefig(join(out_dir, 'ROC_AUC_{}_AllSplits.svg'.format(class_[0])))

        print("AUC for class '{}': {} +/- {}".format(class_[0], np.mean(all_aucs), np.std(all_aucs)))


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
    for class_name, c in list(class_dict.items()):

        pred_out = join(out_dir, 'predictions_{}.csv'.format(class_name))
        labels_out = join(out_dir, 'labels_{}.csv'.format(class_name))

        pred_df = pd.DataFrame(predictions[class_name]).T
        labels_df = pd.DataFrame(labels[class_name]).T

        print(pred_df)
        print(labels_df)

        pred_df.to_csv(pred_out)
        labels_df.to_csv(labels_out)
