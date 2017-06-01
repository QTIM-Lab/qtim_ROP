from learning.retina_net import RetiNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from utils.common import dict_reverse, make_sub_dir
from utils.metrics import confusion_matrix, misclassifications
from plotting import plot_confusion
import numpy as np
from os.path import join, isfile
from visualisation.tsne import tsne
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py

LABELS = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def train_rf(net, train_data):

    # Get CNN codes
    print "Getting features..."
    train_codes = net.predict(train_data)

    # Create random forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
    X_train = train_codes['probabilities']
    y_train = np.asarray(train_codes['y_true'])

    # T-SNE embedding
    # print "T-SNE visualisation of training features"
    # make_tsne(X_train, y_train, out_dir)

    print "Training RF..."
    rf.fit(X_train, y_train)
    return rf, X_train, y_train


def main(model_conf, test_data, raw_images, out_dir, train_data=None):
    """
    
    :param model_conf: YAML file of pre-trained CNN
    :param test_data: HDF5 file of data to test with
    :param raw_images: directory of raw images (referenced in the HDF5 file under 'original_files')
    :param out_dir: output directory for the RF and inference results
    :param train_data: data with which to train the random forest, if not already existing
    :return: ground truth, predictions and computed features for the test data provided
    """

    # Load model and set last layer
    print "Loading trained CNN model..."
    net = RetiNet(model_conf)
    net.set_intermediate('flatten_3')

    # Train/load random forest
    rf_out = join(out_dir, 'rf.pkl')
    train_features_out = join(out_dir, 'cnn_train_features.npy')
    train_labels_out = join(out_dir, 'cnn_train_labels.npy')

    if not isfile(rf_out):
        print "Training new RF on '{}'".format(train_data)
        rf, X_train, y_train = train_rf(net, train_data)
        joblib.dump(rf, rf_out)
        np.save(train_features_out, X_train)
        np.save(train_labels_out, y_train)
    else:
        print "Loading previously trained RF from '{}'".format(rf_out)
        rf = joblib.load(rf_out)
        X_train = np.load(train_features_out)
        y_train = np.load(train_labels_out)

    # Load test data
    print "Extracting test features using pre-trained network '{}'".format(net.experiment_name)
    cnn_features = net.predict(test_data)
    X_test = cnn_features['probabilities']
    y_test = cnn_features['y_true']

    # Make a t-SNE plot combining training and testing
    make_tsne(X_train, y_train, X_test, np.asarray(y_test), out_dir)

    # Save features
    # f = h5py.File(test_data, 'r')
    # index_col = 'filenames' if 'filenames' in f:
    # pd.DataFrame(data=X_test, index=f['filenames']).to_csv(join(out_dir, 'test_features.csv'))

    # Predict classes
    y_pred_classes = rf.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred_classes)
    labels = [k[0] for k in sorted(cnn_features['classes'].items(), key=lambda x: x[1])]
    plot_confusion(confusion, labels, join(out_dir, 'confusion.svg'))

    misclass_dir = make_sub_dir(out_dir, 'misclassified')
    misclassifications(h5py.File(test_data, 'r')['original_files'], raw_images,
                       y_test, y_pred_classes, cnn_features['classes'], misclass_dir)

    # Predict probabilities
    print "Getting RF predictions..."
    y_pred = rf.predict_proba(X_test)

    # col_names = dict_reverse(cnn_features['classes'])
    # roc, thresh, J = roc_auc(y_pred, to_categorical(y_test), col_names, join(out_dir, 'roc_auc.svg'))

    return y_test, y_pred, cnn_features

    # # Confusion matrix, based on best threshold
    # for ci, cn in LABELS.items():
    #     best_thresh = thresh[ci][np.argmax(J[ci])]
    #     print best_thresh
    #     confusion = confusion_matrix(to_categorical(y_test)[:, ci], y_pred[:, ci] > best_thresh)
    #     print confusion
    #     plot_confusion(confusion, ['Not Plus', 'Plus'], join(out_dir, 'confusion_{}.svg'.format(cn)))


def make_tsne(X_train, y_train, X_test, y_test, out_dir, misclassifed=None):

    train_samples = X_train.shape[0]

    saved_tsne = join(out_dir, 'tsne_all.npy')
    if not isfile(saved_tsne):
        X = np.concatenate((X_train, X_test), axis=0)  # combine training and testing for dimensionality reduction
        T = tsne(X, 2, 50, 20.0)  # default parameters for now
        np.save(saved_tsne, T)
    else:
        T = np.load(saved_tsne)

    # Split the training and testing T-SNE, save result
    T_train = T[:train_samples, :]
    T_test = T[train_samples:, :]

    # Plot the training and testing points differently
    pal = itertools.cycle(sns.color_palette('colorblind')[:3])
    fig, ax = sns.plt.subplots()
    for c in (0, 2, 1):
        color = next(pal)
        ax.scatter(T_train[y_train == c, 0], T_train[y_train == c, 1], 20, label=LABELS[c], alpha=0.1, color=color)
        ax.scatter(T_test[y_test == c, 0], T_test[y_test == c, 1], 30, label=LABELS[c], alpha=0.9, color=color)

    ax.legend()
    plt.savefig(join(out_dir, 'tsne_plot.png'))

    # TODO Create a dataframe with the following columns:
    # 'x', 'y', 'z', 'train', 'label', 'misclassified', 'thumbnail'


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-c', '--config', dest='model_config', help="YAML file for model to test", required=True)
    parser.add_argument('-tr', '--train', dest='training_data', help="HDF5 file for training data", default=None)
    parser.add_argument('-te', '--test', dest='test_data', help="HDF5 file for test data", required=True)
    parser.add_argument('-r', '--raw', dest='raw_images', help="Folder of original (raw) images", required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help="Output directory for results", required=True)

    args = parser.parse_args()
    main(args.model_config, args.raw_images, args.test_data, args.out_dir, train_data=args.training_data)
