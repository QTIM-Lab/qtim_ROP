from os.path import isdir, isfile, join, splitext, basename
from qtim_ROP.learning.retina_net import RetiNet
from qtim_ROP.utils.common import dict_reverse, find_images_by_class
from qtim_ROP.utils.image import imgs_by_class_to_th_array
from qtim_ROP.evaluation.metrics import plot_confusion, plot_ROC_curves
from qtim_ROP.deep_rop import generate_report
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
import numpy as np
import h5py
import matplotlib.pyplot as plt

LABELS = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def evaluate(model_config, data, out_dir):

    # Load data
    file_names, img_arr, y_true = load_data(data)
    # y_true = np.asarray([dict_reverse(LABELS)[x] for x in labels])

    y_pred_out = join(out_dir, 'y_pred.npy')
    if not isfile(y_pred_out):

        # Generate model predictions
        model = RetiNet(model_config)
        y_pred = model.predict(img_arr)
        np.save(y_pred_out, y_pred)

    else:
        y_pred = np.load(y_pred_out)

    # Generate report
    labels = [LABELS[i] for i in y_true]
    generate_report(file_names, y_pred, join(out_dir, splitext(basename(model_config))[0]) + '.csv', y_true=labels)

    # Confusion matrix
    arg_max = np.argmax(y_pred, axis=1)
    conf = confusion_matrix(y_true, arg_max)
    plot_confusion(conf, [v for k, v in sorted(list(LABELS.items()), key=lambda x: x[0])], join(out_dir, 'confusion.png'))

    # ROC curves
    fig, ax = plt.subplots()
    plot_ROC_curves(to_categorical(y_true), y_pred, {v: k for k, v in list(LABELS.items()) if v != 'Pre-Plus'})
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig(join(out_dir, 'roc.png'))


def load_data(data):

    if isfile(data):
        f = h5py.File(data)
        return f['original_files'], f['data'], list(f['labels'])
    elif isdir(data):
        return imgs_by_class_to_th_array(data, dict_reverse(LABELS))
    else:
        return None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', dest='model_config', required=True)
    parser.add_argument('-d', '--data', dest='data', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    evaluate(args.model_config, args.data, args.out_dir)
