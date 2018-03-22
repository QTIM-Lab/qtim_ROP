import numpy as np

from ..evaluation.metrics import calculate_metrics
from ..learning.retina_net import RetiNet
from ..utils.common import make_sub_dir


def binary_classifier(model_yaml, test_data, out_dir, merge_disease=True):

    net = RetiNet(model_yaml)

    print("Generating predictions")
    pred_dict = net.evaluate(test_data)

    # Convert three class ground truth to two class
    print("Merging predictions")
    pred_dict['y_true'] = merge_predictions(pred_dict['y_true'], merge_disease=merge_disease)

    # Create new class labels based on how we've merged
    if merge_disease:
        pred_dict['classes'] = {'No': 0, 'Pre-Plus + Plus': 1}
        name = 'PrePlus_Plus'
    else:
        pred_dict['classes'] = {'No + Pre-Plus': 0, 'Plus': 1}
        name = 'No_PrePlus'

    # Convert three class predictions class
    y_pred = merge_predictions(np.argmax(pred_dict['probabilities'], axis=1), merge_disease=merge_disease)

    # Assess performance
    subdir = make_sub_dir(out_dir, name)
    calculate_metrics(pred_dict, y_pred=y_pred, out_dir=subdir)


def merge_predictions(y, merge_disease=True):

    if merge_disease:  # 0: No, 1: Plus, 2: Pre-Plus
        bin_pred = [1 if x in (1, 2) else 0 for x in y]
    else:
        bin_pred = [0 if x in (0, 2) else 1 for x in y]

    return bin_pred

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-c', '--config', dest='model_config', help="YAML file for model to test", required=True)
    parser.add_argument('-t', '--test', dest='test_data', help="HDF5 file for test data", required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help="Output directory for results", required=True)
    parser.add_argument('-m', '--merge-disease', dest='merge_disease', action='store_true', default=False)

    args = parser.parse_args()
    binary_classifier(args.model_config, args.test_data, args.out_dir, merge_disease=args.merge_disease)
