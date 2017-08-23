import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import pandas as pd
import h5py
from os.path import join, isfile
import numpy as np
from qtim_ROP.visualisation.tsne import tsne
from qtim_ROP.learning.retina_net import RetiNet

CLASSES = {0: 'No', 1: 'Plus', 2: 'Pre-Plus'}


def generate_tsne(features, labels, out_file, pal=None):

    X = np.load(features)
    y = np.load(labels)
    tsne_out = join(out_dir, 'tsne.npy')

    if isfile(tsne_out):
        T = np.load(tsne_out)
    else:
        T = tsne(X, 3, 50, 20.0)
        np.save(tsne_out, T)

    # Plot the training and testing points differently
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if pal is None:
        pal = sns.color_palette('colorblind')[:3]
        for c in (0, 2, 1):
            ax.scatter(T[y == c, 0], T[y == c, 1], 30, label=CLASSES[c], alpha=0.7, color=pal[c])
    else:
        for c in (0, 2, 1):
            ax.scatter(T[y == c, 0], T[y == c, 1], 30, label=CLASSES[c], alpha=0.7, color=pal[y == c])
    ax.legend()
    plt.savefig(join(out_file))


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-f', '--features', dest='features', required=True)
    parser.add_argument('-l', '--labels', dest='labels', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    generate_tsne(args.features, args.labels, args.out_dir)
