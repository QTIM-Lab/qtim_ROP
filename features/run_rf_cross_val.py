from os.path import basename, join
from features.rf_cnn_codes import main as cnn_rf
from utils.common import get_subdirs, make_sub_dir, dict_reverse
from utils.metrics import calculate_roc_auc
from keras.utils.np_utils import to_categorical
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_cross_val(all_splits, out_dir):

    all_tpr, all_fpr, all_auc = defaultdict(list), defaultdict(list), defaultdict(list)
    class_dict = None

    for i, split_dir in enumerate(sorted(get_subdirs(all_splits))):

        results_dir = make_sub_dir(out_dir, basename(split_dir))

        cnn_model = join(split_dir, 'Split{}_Model'.format(i), 'Split{0}_Model.yaml'.format(i))
        print cnn_model

        test_data = join(split_dir, 'test.h5')
        y_test, y_pred, cnn_features = cnn_rf(cnn_model, test_data, results_dir)

        roc_auc, fpr, tpr = calculate_roc_auc(y_pred, to_categorical(y_test), cnn_features['classes'], None)

        if not class_dict:
            class_dict = cnn_features['classes']

        for class_name, c in class_dict.items():

            all_tpr[class_name].append(tpr[c])
            all_fpr[class_name].append(fpr[c])
            all_auc[class_name].append(roc_auc[c])

    fig, ax = plt.subplots()
    c_palette = sns.color_palette('colorblind')

    for class_name, c in class_dict.items():

        for n, (fpr, tpr) in enumerate(zip(all_fpr[class_name], all_tpr[class_name])):

            # Mean
            label = 'ROC curve of class {0} (AUC = {1:0.2f}$\pm${1:0.2f} (mean $\pm$ SD)'.format(class_name, np.mean(all_auc[i]), np.std(all_auc[i]))\
                if n == 0 else None
            ax.plot(fpr, tpr, lw=2., c=c_palette[c], label=label)

    plt.savefig(join(out_dir, 'combined_roc.svg'))


if __name__ == '__main__':

    import sys

    run_cross_val(sys.argv[1], sys.argv[2])