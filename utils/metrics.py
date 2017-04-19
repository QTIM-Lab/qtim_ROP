import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from utils.common import dict_reverse, make_sub_dir
from sklearn.metrics import confusion_matrix, classification_report
from plotting import plot_confusion


FONT = {'family': 'sans-serif',
        'color':  'white',
        'weight': 'normal',
        'size': 14}


def calculate_metrics(data_dict, y_pred=None, out_dir=None, ext='.png'):
    predictions, datagen, y_true, class_indices = [data_dict[x] for x in ['probabilities', 'data', 'y_true', 'classes']]

    if y_pred is None:
        y_pred = np.argmax(predictions, axis=1)

    labels = [k[0] for k in sorted(class_indices.items(), key=lambda x: x[1])]
    confusion = confusion_matrix(y_true, y_pred)
    print classification_report(y_true, y_pred)

    # Misclassified images
    misclassified_dir = make_sub_dir(out_dir, 'misclassified')
    misclassifications(datagen.x, y_true, y_pred, class_indices, misclassified_dir)

    # Confusion
    plot_confusion(confusion, labels, join(out_dir, 'confusion' + ext))
    with open(join(out_dir, 'confusion.csv'), 'wb') as conf_csv:
        pd.DataFrame(data=confusion).to_csv(conf_csv)


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