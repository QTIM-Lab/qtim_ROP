import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.misc import imresize
import seaborn as sns
sns.set_style('ticks')
import pandas as pd
import h5py
from os.path import join, isfile
import numpy as np
from qtim_ROP.visualisation.tsne import tsne
from qtim_ROP.learning.retina_net import RetiNet

CLASSES = {0: 'Normal', 1: 'Plus', 2: 'Pre-Plus'}


def generate_tsne(features, labels, out_dir, skip=3, pal=None):

    X = np.load(features)[::skip]
    y = np.load(labels)[::skip]
    tsne_out = join(out_dir, 'tsne.npy')

    if isfile(tsne_out):
        T = np.load(tsne_out)
    else:
        T = tsne(X, 3, 50, 20.0)
        np.save(tsne_out, T)

    # Plot the training and testing points differently
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ['o', 's', '^']

    if pal is None:
        pal = sns.color_palette('colorblind')[:3]
        for c in (0, 2, 1):
            ax.scatter(T[y == c, 0], T[y == c, 1], 40, label=CLASSES[c], alpha=0.7, color=pal[c])
    else:
        pal = pal[::skip]
        for c in (0, 2, 1):
            ax.scatter(T[y == c, 0], T[y == c, 1], 40, label=CLASSES[c], marker=markers[c], alpha=0.9, color=pal[y == c], edgecolors='gray')

    ax.legend()
    ax.set_ylim([-120, 120])
    plt.savefig(join(out_dir, 'tsne_plot.png'), dpi=300)

    return T


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-f', '--features', dest='features', required=True)
    parser.add_argument('-l', '--labels', dest='labels', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    args = parser.parse_args()
    generate_tsne(args.features, args.labels, args.out_dir)
