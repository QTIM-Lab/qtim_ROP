import pandas as pd
from os.path import join, basename

import numpy as np
from scipy.stats import mode

from ..evaluation.metrics import calculate_metrics
from ..learning.retina_net import RetiNet
from ..utils.common import get_subdirs, make_sub_dir

OPTIONS = ('vote', 'average', 'confidence', 'weighted_average', 'merge')


class ReaderEnsemble(object):

    def __init__(self, models_dir, out_dir, names=None, modes=OPTIONS):

        self.models_dir = models_dir
        self.models_list = get_subdirs(models_dir) if not names else [join(models_dir, name) for name in names]

        print('Loading models')
        self.models = [RetiNet(join(model_dir, basename(model_dir) + '.yaml')) for model_dir in self.models_list]

        if any(x not in OPTIONS for x in modes):
            raise ValueError("Invalid option(s) '{}'\nChoose one or more from '()', '()', '()'"
                             .format(modes,*OPTIONS))
        self.modes = modes
        self.out_dir = out_dir

    def evaluate(self, test_data):

        all_probs, all_votes, all_data = [], [], []

        for model in self.models:

            # Predict class probabilities using current model
            print("Generating predictions for model '{}".format(model.experiment_name))
            data_dict = model.evaluate(test_data)

            pred_prob = data_dict['probabilities']
            all_probs.append(pred_prob)  # keep probabilities
            all_votes.append(np.argmax(pred_prob, axis=1))  # keep class predictions
            all_data.append(data_dict)

            # Calculate metrics
            test_dir = make_sub_dir(self.out_dir, model.experiment_name)
            calculate_metrics(data_dict, out_dir=test_dir)

            # Save predictions
            pd.DataFrame(np.asarray(pred_prob)).to_csv(join(self.out_dir, '{}.csv'.format(model.experiment_name)))

        # Test ensemble(s)
        data = all_data[0]

        if 'vote' in self.modes:

            vote_dir = make_sub_dir(self.out_dir, 'vote')
            vote = majority_vote(all_votes)
            calculate_metrics(data, y_pred=vote, out_dir=vote_dir)

        if 'average' in self.modes:

            average_dir = make_sub_dir(self.out_dir, 'average')
            average = average_probabilities(all_probs)
            calculate_metrics(data, y_pred=average, out_dir=average_dir)

        if 'confidence' in self.modes:

            conf_dir = make_sub_dir(self.out_dir, 'confidence')
            conf = max_prob(all_probs)
            calculate_metrics(data, y_pred=conf, out_dir=conf_dir)


def majority_vote(votes):

    mode_result = mode(votes, axis=0)
    return mode_result.mode[0]


def average_probabilities(probs):

    return np.argmax(np.mean(probs, axis=0), axis=1)


def max_prob(probs):

    mc = []
    for i in range(0, probs[0].shape[0]):

        readers = [probs[r][i] for r in range(0, len(probs))]  # probabilties from all readers for this image
        most_confident = np.argmax(np.max(readers, axis=1))  # the index of the most confident reader
        mc.append(np.argmax(readers[most_confident]))  # the predicted label of the most confident reader

    return np.asarray(mc)


def combine_models(models):
    raise NotImplementedError('Model weight averaging not implemented yet!')


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--models-dir', dest='models_dir', help="Directory where models are kept", required=True)
    parser.add_argument('-t', '--test-data', dest='test_data', help="HD5 file with fields 'data' and 'labels'", required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', help="Directory to output results", required=True)
    parser.add_argument('-n', '--names', dest='names', help="Relative paths of models to ensemble", nargs='+', required=False, default=None)


    args = parser.parse_args()
    ensemble = ReaderEnsemble(args.models_dir, args.out_dir, names=args.names)
    ensemble.evaluate(args.test_data)
