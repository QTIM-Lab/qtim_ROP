from os.path import basename, join
from features.rf_cnn_codes import main as cnn_rf
from utils.common import get_subdirs, make_sub_dir


def run_cross_val(all_splits, out_dir):

    all_ground_truth, all_predictions = [], []

    for i, split_dir in enumerate(sorted(get_subdirs(all_splits))):

        results_dir = make_sub_dir(out_dir, basename(split_dir))

        cnn_model = join(split_dir, 'Split{}_Model'.format(i), 'Split{0}_Model.yaml'.format(i))
        print cnn_model

        test_data = join(split_dir, 'test.h5')
        y_test, y_pred, _ = cnn_rf(cnn_model, test_data, results_dir)

        all_ground_truth.append(y_test)
        all_predictions.append(y_pred)


if __name__ == '__main__':

    import sys

    run_cross_val(sys.argv[1], sys.argv[2])