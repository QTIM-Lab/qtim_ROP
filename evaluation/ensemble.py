from utils.common import make_sub_dir, get_subdirs
from utils.image import imgs_by_class_to_th_array
from metrics import confusion
from learning.retina_rf import RetinaRF, locate_config
from os.path import join
from glob import glob
import numpy as np

CLASS_LABELS = {'No': 0, 'Plus': 1, 'Pre-Plus': 2}


def evaluate_ensemble(models_dir, test_images, out_dir):

    # Get images and true classes
    img_arr, y_true = imgs_by_class_to_th_array(test_images, CLASS_LABELS)
    print img_arr.shape

    y_pred_ensemble = []

    # Load each model
    for i, model_dir in enumerate(get_subdirs(models_dir)):

        # Load model
        print "Loading CNN/RF #{}".format(i)
        model_config, rf_pkl = locate_config(model_dir)
        cnn_rf = RetinaRF(model_config, rf_pkl=rf_pkl)

        # Predicted probabilities
        print "Making predictions..."
        y_preda = cnn_rf.predict(img_arr)
        y_pred_ensemble.append(y_preda)

        y_pred = np.argmax(y_preda, axis=1)
        confusion(y_true, y_pred, CLASS_LABELS, join(out_dir, 'confusion_{}.png'.format(i)))

    # Evaluate ensemble
    y_pred_ensemble = np.mean(np.dstack(y_pred_ensemble), axis=2)
    confusion(y_true, y_pred_ensemble, CLASS_LABELS, join(out_dir, 'confusion_ensemble.png'))


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('-m', '--models', dest='models', help='Folder of models', required=True)
    parser.add_argument('-i', '--images', dest='images', help='Images to test', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)

    args = parser.parse_args()

    evaluate_ensemble(args.models, args.images, args.out_dir)

