#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from ..utils.common import *


def analyse_vessels(orig_dir, seg_dir, out_dir, thresh):

    # Find images
    orig_dict = find_images_by_class(orig_dir)
    seg_dict = find_images_by_class(seg_dir)

    assert(all(len(orig_dict[c]) == len(seg_dict[c]) for c in list(orig_dict.keys())))

    pixel_totals, pixel_hist = {}, {}

    heights = [480] * 3
    widths = [640] * 10

    fig_width = 18.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    font = {'family': 'sans-serif', 'color': 'white', 'weight': 'bold', 'size': 16}

    fig1, ax1 = plt.subplots(3, 10, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': heights})
    fig2, ax2 = plt.subplots(3, 10, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': heights})
    fig3, ax3 = plt.subplots(3, 10, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': heights})

    cidx = 0

    for (class_, orig_list), (_, seg_list) in zip(list(orig_dict.items()), list(seg_dict.items())):

        # Load all original images in this class
        print("Loading original '{}' images".format(class_))
        orig_images = np.array([np.asarray(Image.open(im)) for im in orig_list])

        # Load segmented images and apply threshold
        print("Loading segmented '{}' images".format(class_))
        seg_images = np.array([np.asarray(Image.open(im)) for im in seg_list])

        # Compute total pixels per image
        vessel_pixels = np.sum(seg_images > (thresh * 255.0), axis=(1, 2))
        pixel_totals[class_] = vessel_pixels

        # Sort the original images by total segmented vessel pixels
        order = np.argsort(vessel_pixels)

        sorted_orig = orig_images[order]
        sorted_seg = seg_images[order]
        sorted_names = [basename(orig_list[x]) for x in order]

        print("Original images: {}".format(sorted_orig.shape))
        print("Segmented images: {}".format(sorted_seg.shape))

        step = int(np.ceil(sorted_orig.shape[0] / 10.0))

        fewest_dir = join(out_dir, class_ + '_fewest')
        if not isdir(fewest_dir):
            mkdir(fewest_dir)

        for j in range(0, step):
            im_name = sorted_names[j]
            Image.fromarray(sorted_orig[j]).save(join(fewest_dir, '{}.jpg'.format(im_name)))

        sample = list(range(0, sorted_orig.shape[0], step))

        for i, idx in enumerate(sample):

            orig = Image.fromarray(sorted_orig[idx])
            seg = np.invert(Image.fromarray(sorted_seg[idx]))

            bin = (sorted_seg[idx] > (thresh * 255.0)).astype(np.uint8) * 255
            bin = Image.fromarray(np.dstack([bin] * 3))

            if i == 0:
                ax1[cidx, i].text(1., 1., class_, fontdict=font, verticalalignment='top')
                ax2[cidx, i].text(1., 1., class_, fontdict=font, verticalalignment='top')
                ax3[cidx, i].text(1., 1., class_, fontdict=font, verticalalignment='top')

            ax1[cidx, i].imshow(orig)
            ax1[cidx, i].axis('off')

            ax2[cidx, i].imshow(seg)
            ax2[cidx, i].axis('off')

            ax3[cidx, i].imshow(bin)
            ax3[cidx, i].axis('off')

        cidx += 1

    fig1.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    fig1.savefig(join(out_dir, 'orig_order.png'))

    fig2.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    fig2.savefig(join(out_dir, 'seg_order.png'))

    fig3.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    fig3.savefig(join(out_dir, 'bin_order.png'))

    # Plot histogram
    df = pd.DataFrame({k: pd.Series(v) for k, v in pixel_totals.items()})
    fig, ax = plt.subplots()
    cols = ['r', 'g', 'b']

    for (class_, total_pixels), color in zip(list(pixel_totals.items()), cols):
        plt.hist(total_pixels, normed=True, stacked=True, color=color, alpha=0.25, label=class_)

    plt.legend(list(pixel_totals.keys()))
    plt.xlabel('Number of vessel pixels')
    plt.ylabel('Normalized frequency')
    plt.savefig(join(out_dir, 'hist.png'))

    # Box plot
    fig, ax = plt.subplots()
    sns.boxplot(data=df, order=['No', 'Pre-Plus', 'Plus'])
    sns.plt.xlabel('Class')
    sns.plt.ylabel('Total vessel pixels')
    sns.plt.savefig(join(out_dir, 'boxplot.png'))

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='in_dir', help='Folder of original images', required=True)
    parser.add_argument('-s', '--segmented', dest='seg_dir', help='Folder of segmented images', required=True)
    parser.add_argument('-o', '--output', dest='out_dir', help='Output folder', required=True)
    parser.add_argument('-t', '--threshold', dest='thresh', help='Threshold value', type=float, required=True)

    args = parser.parse_args()
    analyse_vessels(args.in_dir, args.seg_dir, args.out_dir, args.thresh)
