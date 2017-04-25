import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from utils.common import dict_reverse, make_sub_dir
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.utils.np_utils import to_categorical
from plotting import plot_confusion
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, auc
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

    # ROC/AUC
    roc_auc(pred_df, true_df, join(out_dir, 'roc_auc' + ext))

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


# Predict data
def misclassifications(data, y_true, y_pred, classes, out_dir, n=10):

    class_count = [0] * (np.max(y_true) + 1)
    fig, ax = plt.subplots()

    classes = dict_reverse(classes)

    for img, yt, yp in zip(data, y_true, y_pred):

        if yt != yp and class_count[yp] < n:

            plt.cla()
            ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.text(5, 10, 'True: {}'.format(classes[yt]), fontdict=FONT)
            ax.text(5, 25, 'Predicted: {}'.format(classes[yp]), fontdict=FONT)
            ax.axis('off')

            plt.savefig(join(out_dir, '{}_{}.png'.format(classes[yp], class_count[yp])))
            class_count[yp] += 1


def roc_auc(predictions, y_true, out_path):

    col_names = {k: v for k, v in enumerate(y_true.columns)}
    n_classes = len(col_names)

    y_pred = predictions.as_matrix()
    y_true = y_true.as_matrix()

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-averaging
    fpr["micro"], tpr["micro"], thresholds = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-averaging
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 1.
    fig, ax = plt.subplots()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-averaging (AUC = {0:0.2f})'
             .format(roc_auc["micro"]), linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"], label='macro-averaging (AUC = {0:0.2f})'
             .format(roc_auc["macro"]), linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip([0, 2, 1], colors):
    #     plt.plot(fpr[i], tpr[i], lw=lw,
    #              label='ROC curve of class {0} (AUC = {1:0.2f})'
    #              ''.format(col_names[i], roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.xlim([-0.025, 1.025])
    ax.ylim([-0.025, 1.025])
    ax.xlabel('False Positive Rate')
    ax.ylabel('True Positive Rate')
    ax.title('ROC/AUC')
    ax.legend(loc="lower right")
    plt.savefig(out_path)