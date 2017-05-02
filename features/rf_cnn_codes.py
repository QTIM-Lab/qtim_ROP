from learning.retina_net import RetiNet
from keras.utils.np_utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from utils.common import dict_reverse, make_sub_dir
from utils.metrics import confusion_matrix, misclassifications
from plotting import plot_confusion
import numpy as np
from os.path import join
from tsne import tsne
import matplotlib.pyplot as plt
import pandas as pd
import h5py

LABELS = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def train_rf(net, train_data, out_dir):

    # Get CNN codes
    print "Getting features..."
    train_codes = net.predict(train_data)

    # Create random forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
    X_train = train_codes['probabilities']
    y_train = np.asarray(train_codes['y_true'])

    # T-SNE embedding
    print "T-SNE visualisation of training features"
    np.save(join(out_dir, 'cnn_train_features.npy'), X_train)
    make_tsne(X_train, y_train, out_dir)

    print "Training RF..."
    rf.fit(X_train, y_train)
    return rf


def main(model_conf, test_data, out_dir, train_data=None):

    # Load model and set last layer
    print "Loading model..."
    net = RetiNet(model_conf)
    net.set_intermediate('flatten_3')

    # Train/load random forest
    rf_out = join(out_dir, 'rf.pkl')
    if train_data is not None:
        print "Training new RF on '{}'".format(train_data)
        rf = train_rf(net, train_data, out_dir)
        joblib.dump(rf, rf_out)
    else:
        print "Loading previously trained RF from '{}'".format(rf_out)
        rf = joblib.load(rf_out)

    # Load test data
    print "Getting CNN features using pre-trained network '{}'".format(net.experiment_name)
    cnn_features = net.predict(test_data)
    X_test = cnn_features['probabilities']
    y_test = cnn_features['y_true']

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
    misclassifications(h5py.File(test_data, 'r')['data'], y_test, y_pred_classes, cnn_features['classes'], misclass_dir)

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


def make_tsne(X, y, out_dir):

    T = tsne(X, 2, 50, 20.0)
    fig, ax = plt.subplots()
    ax.scatter(T[y == 0, 0], T[y == 0, 1], 20, label=LABELS[0], alpha=0.6)
    ax.scatter(T[y == 1, 0], T[y == 1, 1], 20, label=LABELS[1], alpha=0.6)
    ax.scatter(T[y == 2, 0], T[y == 2, 1], 20, label=LABELS[2], alpha=0.6)
    ax.legend()
    plt.savefig(join(out_dir, 'tsne.png'))


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-c', '--config', dest='model_config', help="YAML file for model to test", required=True)
    parser.add_argument('-tr', '--train', dest='training_data', help="HDF5 file for training data", default=None)
    parser.add_argument('-te', '--test', dest='test_data', help="HDF5 file for test data", required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help="Output directory for results", required=True)


    args = parser.parse_args()
    main(args.model_config,  args.test_data, args.out_dir, train_data=args.training_data)
