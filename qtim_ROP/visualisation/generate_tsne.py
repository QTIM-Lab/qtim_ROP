import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, isfile
import numpy as np
from tsne import tsne

CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def generate_tsne(features, labels, out_dir):

    X = np.load(features)
    y = np.load(labels)
    tsne_out = join(out_dir, 'tsne.npy')

    if isfile(tsne_out):
        T = np.load(tsne_out)
    else:
        T = tsne(X, 3, 50, 20.0)
        np.save(tsne_out, T)

    # Plot the training and testing points differently
    pal = sns.color_palette('colorblind')[:3]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for c in (0, 2, 1):
        ax.scatter(T[y == c, 0], T[y == c, 1], 30, label=CLASSES[c], alpha=0.7, color=pal[c])

    ax.legend()
    plt.savefig(join(out_dir, 'tsne_plot.png'))


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-f', '--features', dest='features', required=True)
    parser.add_argument('-l', '--labels', dest='labels', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    generate_tsne(args.features, args.labels, args.out_dir)
