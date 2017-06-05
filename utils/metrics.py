import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from utils.common import dict_reverse, make_sub_dir
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, accuracy_score
from keras.utils.np_utils import to_categorical
from plotting import plot_confusion
from PIL import Image
from itertools import cycle
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy import interp

FONT = {'family': 'sans-serif',
        'color':  'white',
        'weight': 'normal',
        'size': 14}


def calculate_metrics(data_dict, out_dir, y_pred=None, ext='.png'):

    # Get useful information from data dictionary
    predictions, datagen, y_true, class_indices = [data_dict[x] for x in ['probabilities', 'data', 'y_true', 'classes']]

    # Create data frames for predictions and ground truth
    cols = np.asarray(sorted([[k, v] for k, v in class_indices.items()], key=lambda x: x[1]))
    pred_df = pd.DataFrame(data=predictions, columns=cols[:, 0])
    true_df = pd.DataFrame(data=to_categorical(y_true), columns=cols[:, 0])
    pred_df.to_csv(join(out_dir, 'predictions.csv'))
    true_df.to_csv(join(out_dir, 'ground_truth.csv'))

    # For the sake of calculating confusion matrices
    if y_pred is None:
        y_pred = np.argmax(predictions, axis=1)

    # Confusion
    labels = [k[0] for k in sorted(class_indices.items(), key=lambda x: x[1])]
    confusion = confusion_matrix(y_true, y_pred)

    plot_confusion(confusion, labels, join(out_dir, 'confusion' + ext))
    with open(join(out_dir, 'confusion.csv'), 'wb') as conf_csv:
        pd.DataFrame(data=confusion).to_csv(conf_csv)

    print "Accuracy: {}".format(accuracy_score(y_true, y_pred))
    print classification_report(y_true, y_pred)

    # Misclassified images  #  TODO fix bug when classes < 3
    misclassified_dir = make_sub_dir(out_dir, 'misclassified')
    misclassifications(datagen.x, y_true, y_pred, class_indices, misclassified_dir)

    # ROC/AUC
    col_names = {k: v for k, v in enumerate(y_true.columns)}
    y_pred = predictions.as_matrix()
    y_true = y_true.as_matrix()
    calculate_roc_auc(y_pred, y_true, col_names, join(out_dir, 'roc_auc' + ext))


def misclassifications(file_names, img_path, y_true, y_pred, classes, out_dir, n=10):

    class_count = [0] * (np.max(y_true) + 1)
    fig, ax = plt.subplots()

    classes = dict_reverse(classes)

    for img, yt, yp in zip(file_names, y_true, y_pred):

        if yt != yp:  # and class_count[yp] < n:

            plt.cla()
            img = np.asarray(Image.open(join(img_path, img)))
            ax.imshow(img)  # np.transpose(img, (1, 2, 0)))
            ax.text(5, 10, 'True: {}'.format(classes[yt]), fontdict=FONT)
            ax.text(5, 25, 'Predicted: {}'.format(classes[yp]), fontdict=FONT)
            ax.axis('off')

            plt.savefig(join(out_dir, '{}_{}.png'.format(classes[yp], class_count[yp])))
            class_count[yp] += 1


def plot_ROC_splits(y_true_all, y_pred_all, (class_name, class_index)):

    line_styles = cycle(['-', '--', '-.', ':', 'solid'])

    for split_no, (y_true_split, y_pred_split) in enumerate(zip(y_true_all, y_pred_all)):

        fpr, tpr, thresholds = roc_curve(y_true_split[:, class_index], y_pred_split[:, class_index])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='CV #{}, AUC = {:.3f}'.format(split_no + 1, roc_auc), linestyle=next(line_styles))

    plt.title('Receiver operating characteristic for "{}" class'.format(class_name))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


def plot_PR_splits(y_true_all, y_pred_all, (class_name, class_index)):

    line_styles = cycle(['-', '--', '-.', ':', 'solid'])

    for split_no, (y_true_split, y_pred_split) in enumerate(zip(y_true_all, y_pred_all)):
        precision, recall, thresholds = precision_recall_curve(y_true_split[:, class_index], y_pred_split[:, class_index])
        pr_auc = average_precision_score(y_true_split[:, class_index], y_pred_split[:, class_index])

        plt.plot(recall, precision, label='CV #{}, AUC = {:.3f}'.format(split_no + 1, pr_auc), linestyle=next(line_styles))

    plt.title('Precision recall curves for "{}" class'.format(class_name))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


def plot_ROC_by_class(y_true, y_pred, classes, ls='-'):

    best_thresh = {}
    for class_name, c in classes.items():  # for each class

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true[:, c], y_pred[:, c])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(class_name, roc_auc), linestyle=ls)

        # Calculate J statistic
        J = [j_statistic(y_true[:, c], y_pred[:, c], t) for t in thresholds]
        j_best = np.argmax(J)

        # Store best threshold for each class
        best_thresh[class_name] = J[j_best]

    return best_thresh


def plot_PR_by_class(y_pred, y_true, classes, out_path):

    best_thresh = {}
    for class_name, c in classes.items():  # for each class

        # Compute ROC curve
        precision, recall, thresholds = precision_recall_curve(y_true[:, c], y_pred[:, c])
        pr_auc = auc(recall, precision)

        # Plot PR curve
        plt.plot(recall, precision, label='{}, AUC = {:.3f}'.format(class_name, pr_auc))

        # Calculate J statistic
        J = [j_statistic(y_true, y_pred, t) for t in thresholds]
        j_best = np.argmax(J)

        # Store best threshold for each class
        best_thresh[class_name] = J[j_best]

    return best_thresh


# def plot_roc_auc(predictions, ground_truth):
#
#     # Predictions and ground truth as numpy arrays
#     y_pred = np.asarray(predictions).ravel()
#     y_true = np.asarray(ground_truth).ravel()
#
#     # Get ROC curves
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#
#     # Return index of best model by J statistic
#
#
#     return thresholds[Ji], fpr[Ji], tpr[Ji]


def j_statistic(y_true, y_pred, thresh):

    C = confusion_matrix(y_true, y_pred > thresh)
    TN = C[0, 0]
    FN = C[1, 0]
    TP = C[1, 1]
    FP = C[0, 1]

    j = (TP / float(TP + FN)) + (TN / float(TN + FP)) - 1
    return j


# def calculate_roc_auc(y_pred, y_true, col_names, out_path):
#
#     n_classes = len(col_names)
#
#     fpr, tpr, roc_auc, thresh, J = {}, {}, {}, {}, {}
#     for i in range(n_classes):
#         fpr[i], tpr[i], thresh[i] = roc_curve(y_true[:, i], y_pred[:, i])
#         J[i] = tpr[i] + (1 - fpr[i]) - 1
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     return roc_auc, fpr, tpr
#
#     # Micro-averaging
#     fpr["micro"], tpr["micro"], thresholds = roc_curve(y_true.ravel(), y_pred.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     # Macro-averaging
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
#     # Plot all ROC curves
#     lw = 1.
#     fig, ax = plt.subplots()
#     plt.plot(fpr["micro"], tpr["micro"], label='micro-averaging (AUC = {0:0.2f})'
#              .format(roc_auc["micro"]), linestyle=':', linewidth=4)
#
#     plt.plot(fpr["macro"], tpr["macro"], label='macro-averaging (AUC = {0:0.2f})'
#              .format(roc_auc["macro"]), linestyle=':', linewidth=4)
#
#     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#     for i, color in zip([0, 1], colors):
#         plt.plot(fpr[i], tpr[i], lw=lw,
#                  label='ROC curve of class {0} (AUC = {1:0.2f})'
#                  ''.format(col_names[i], roc_auc[i]))
#
#     ax.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([-0.025, 1.025])
#     plt.ylim([-0.025, 1.025])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC/AUC')
#     plt.legend(loc="lower right")
#     plt.savefig(out_path)
#
#     return roc_auc, thresh, J