import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from utils.common import dict_reverse

FONT = {'family': 'sans-serif',
        'color':  'white',
        'weight': 'normal',
        'size': 14}

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