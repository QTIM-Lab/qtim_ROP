import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from ..utils.common import dict_reverse, make_sub_dir
from ..utils.plotting import plot_confusion
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, accuracy_score
from keras.utils.np_utils import to_categorical
from PIL import Image
from itertools import cycle
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import seaborn as sns
sns.set_style('ticks')

FONT = {'family': 'sans-serif',
        'color':  'white',
        'weight': 'normal',
        'size': 14}


def calculate_metrics(data_dict, out_dir, y_pred=None, ext='.png'):

    # Get useful information from data dictionary
    predictions, datagen, y_true, class_indices = [data_dict[x] for x in ['probabilities', 'data', 'y_true', 'classes']]

    # Create data frames for predictions and ground truth
    cols = np.asarray(sorted([[k, v] for k, v in list(class_indices.items())], key=lambda x: x[1]))
    pred_df = pd.DataFrame(data=predictions, columns=cols[:, 0])
    true_df = pd.DataFrame(data=to_categorical(y_true), columns=cols[:, 0])
    pred_df.to_csv(join(out_dir, 'predictions.csv'))
    true_df.to_csv(join(out_dir, 'ground_truth.csv'))

    # For the sake of calculating confusion matrices
    if y_pred is None:
        y_pred = np.argmax(predictions, axis=1)

    # Confusion
    labels = [k[0] for k in sorted(list(class_indices.items()), key=lambda x: x[1])]
    confusion = confusion_matrix(y_true, y_pred)

    plot_confusion(confusion, labels, join(out_dir, 'confusion' + ext))
    with open(join(out_dir, 'confusion.csv'), 'wb') as conf_csv:
        pd.DataFrame(data=confusion).to_csv(conf_csv)

    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    print(classification_report(y_true, y_pred))

    # Misclassified images  #  TODO fix bug when classes < 3
    misclassified_dir = make_sub_dir(out_dir, 'misclassified')
    misclassifications(datagen.x, y_true, y_pred, class_indices, misclassified_dir)

    # ROC/AUC
    col_names = {k: v for k, v in enumerate(y_true.columns)}
    y_pred = predictions.as_matrix()
    y_true = y_true.as_matrix()

    fig, ax = plt.subplots()
    plot_ROC_curves(y_pred, y_true, col_names)
    plt.savefig(join(out_dir, 'roc_auc' + ext))


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


def plot_ROC_splits(y_true_all, y_pred_all, xxx_todo_changeme):

    (class_name, class_index) = xxx_todo_changeme
    line_styles = cycle(['-', '--', '-.', ':', 'solid'])
    all_aucs = []

    for split_no, (y_true_split, y_pred_split) in enumerate(zip(y_true_all, y_pred_all)):

        fpr, tpr, thresholds = roc_curve(y_true_split[:, class_index], y_pred_split[:, class_index])
        roc_auc = auc(fpr, tpr)
        all_aucs.append(roc_auc)

        plt.plot(fpr, tpr, label='Split #{}, AUC = {:.3f}'.format(split_no + 1, roc_auc), linestyle=next(line_styles))

    plt.title('Cross-validated ROC curves for {}'.format(class_name))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    return all_aucs


def plot_PR_splits(y_true_all, y_pred_all, xxx_todo_changeme1):

    (class_name, class_index) = xxx_todo_changeme1
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


def plot_ROC_curves(y_true, y_pred, classes, ls='-', regression=False, outfile=None):

    # best_thresh = {}
    aucs = []

    for c, class_name in list(classes.items()):  # for each class

        # Compute ROC curve
        if regression:
            fpr, tpr, thresholds = roc_curve(y_true == c, y_pred)
            if c == 0:
                fpr = 1 - fpr
                tpr = 1 - tpr
        else:
            fpr, tpr, thresholds = roc_curve(y_true[:, c], y_pred[:, c])
            # J = [j_statistic(y_true[:, c], y_pred[:, c], t) for t in thresholds]
            # j_best = np.argmax(J)
            # best_thresh[class_name] = J[j_best]

        roc_auc = auc(fpr, tpr)
        aucs.append(dict(method='ROC', label=class_name, score=roc_auc))

        # Plot ROC curve
        plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(class_name, roc_auc), linestyle=ls)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if outfile:
        plt.savefig(outfile, dpi=300)

    return aucs


def plot_PR_curves(y_true, y_pred, classes, outfile):

    plt.clf()
    aucs = []
    for c, class_name in list(classes.items()):  # for each class

        # Compute ROC curve
        precision, recall, _ = precision_recall_curve(y_true[:, c], y_pred[:, c])
        pr_auc = auc(recall, precision)
        aucs.append(dict(method='PR', label=class_name, score=pr_auc))

        # Plot PR curve
        plt.plot(recall, precision, label='{}, AUC = {:.3f}'.format(class_name, pr_auc))

    plt.legend(loc='upper right')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.ylabel('Precision')
    plt.xlabel('Recall')

    if outfile:
        plt.savefig(outfile, dpi=300)

    return aucs


def confusion(y_true, y_pred, classes, out_path):

    confusion = confusion_matrix(y_true, y_pred)
    labels = [k[0] for k in sorted(list(classes.items()), key=lambda x: x[1])]
    plot_confusion(confusion, labels, out_path)


def j_statistic(y_true, y_pred, thresh):

    C = confusion_matrix(y_true, y_pred > thresh)
    TN = C[0, 0]
    FN = C[1, 0]
    TP = C[1, 1]
    FP = C[0, 1]

    j = (TP / float(TP + FN)) + (TN / float(TN + FP)) - 1
    return j


def fpr_and_tpr(y_true, y_pred):

    C = confusion_matrix(y_true, y_pred)
    TN = C[0, 0]
    FN = C[1, 0]
    TP = C[1, 1]
    FP = C[0, 1]

    tpr = TP / float(TP + FN)
    fpr = FP / float(FP + TN)

    return fpr, tpr

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