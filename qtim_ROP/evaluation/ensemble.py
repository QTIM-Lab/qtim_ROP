from ..utils.common import make_sub_dir, get_subdirs
from ..utils.image import imgs_by_class_to_th_array
from ..evaluation.metrics import confusion
from ..learning.retina_net import RetiNet, RetinaRF, locate_config
from os.path import join, exists
from glob import glob
import numpy as np
from scipy.stats import mode
from sklearn.metrics import cohen_kappa_score

CLASS_LABELS = {'No': 0, 'Plus': 1, 'Pre-Plus': 2}


def evaluate_ensemble(models_dir, test_images, out_dir, rf=False):

    # Get images and true classes
    img_arr, y_true = imgs_by_class_to_th_array(test_images, CLASS_LABELS)
    print(img_arr.shape)

    y_pred_all = []

    # Load each model
    for i, model_dir in enumerate(get_subdirs(models_dir)):

        # Load model
        if rf:
            print("Loading CNN+RF #{}".format(i))
            model_config, rf_pkl = locate_config(model_dir)
            model = RetinaRF(model_config, rf_pkl=rf_pkl)
        else:
            print("Loading CNN #{}".format(i))
            config_file = glob(join(model_dir, '*.yaml'))[0]
            model = RetiNet(config_file).model

        # Predicted probabilities
        print("Making predictions...")
        ypred_out = join(out_dir, 'ypred_{}.npy'.format(i))

        if not exists(ypred_out):
            y_preda = model.predict(img_arr)
            np.save(ypred_out, y_preda)
        else:
            y_preda = np.load(ypred_out)

        y_pred_all.append(y_preda)
        y_pred = np.argmax(y_preda, axis=1)

        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        confusion(y_true, y_pred, CLASS_LABELS, join(out_dir, 'confusion_split{}_k={:.3f}.png'.format(i, kappa)))

    # Evaluate ensemble
    y_preda_ensemble = np.mean(np.dstack(y_pred_all), axis=2)
    y_pred_ensemble = np.argmax(y_preda_ensemble, axis=1)
    kappa = cohen_kappa_score(y_true, y_pred_ensemble)
    confusion(y_true, y_pred_ensemble, CLASS_LABELS, join(out_dir, 'confusion_ensemble_k={:.3f}.png'.format(kappa)))


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('-m', '--models', dest='models', help='Folder of models', required=True)
    parser.add_argument('-i', '--images', dest='images', help='Images to test', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help='Output directory', required=True)
    parser.add_argument('-r', '--rf', action='store_true', dest='rf', help='Use random forest on CNN features?',
                        default=False)
    args = parser.parse_args()

    evaluate_ensemble(args.models, args.images, args.out_dir, rf=args.rf)

