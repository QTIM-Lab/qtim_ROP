from os.path import join, basename
import numpy as np
from scipy.stats import mode
from learning.retina_net import RetiNet
from utils.common import get_subdirs, make_sub_dir
from utils.metrics import calculate_metrics

OPTIONS = ('vote', 'average', 'weighed_average', 'merge')


class ReaderEnsemble(object):

    def __init__(self, models_dir, out_dir, names=None, modes=OPTIONS):

        self.models_dir = models_dir
        self.models_list = get_subdirs(models_dir) if not names else [join(models_dir, name) for name in names]

        print 'Loading models'
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
            print "Generating predictions for model '{}".format(model.experiment_name)
            data_dict = model.predict(test_data)

            pred_prob = data_dict['probabilities']
            all_probs.append(pred_prob)  # keep probabilities
            all_votes.append(np.argmax(pred_prob, axis=1))  # keep class predictions
            all_data.append(data_dict)

            # Calculate metrics
            test_dir = make_sub_dir(self.out_dir, model.experiment_name)
            calculate_metrics(data_dict, out_dir=test_dir)

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


def majority_vote(votes):

    mode_result = mode(votes, axis=0)
    return mode_result.mode[0]


def average_probabilities(probs):

    return np.argmax(np.mean(probs, axis=0), axis=1)


def combine_models(models):
    raise NotImplementedError('Model weight averaging not implemented yet!')


if __name__ == '__main__':

    import sys
    ensemble = ReaderEnsemble(sys.argv[1], sys.argv[2], names=['Reader1_Seg1', 'Reader2_Seg1', 'Reader3_Seg1'])
    ensemble.evaluate(sys.argv[3])
