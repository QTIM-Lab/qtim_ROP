#!/usr/bin/env python


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from common import *


def analyse_vessels(orig_dir, seg_dir, out_dir, thresh):

    # Find images
    orig_dict = find_images_by_class(orig_dir)
    seg_dict = find_images_by_class(seg_dir)

    assert(all(len(orig_dict[c]) == len(seg_dict[c]) for c in orig_dict.keys()))

    pixel_totals, pixel_hist = {}, {}

    heights = [480] * 3
    widths = [640] * 10

    fig_width = 18.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    font = {'family': 'sans-serif', 'color': 'white', 'weight': 'bold', 'size': 16}

    fig1, ax1 = plt.subplots(3, 10, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': heights})
    fig2, ax2 = plt.subplots(3, 10, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': heights})
    cidx = 0

    for (class_, orig_list), (_, seg_list) in zip(orig_dict.items(), seg_dict.items()):

        # Load all original images in this class
        orig_images = np.array([np.asarray(Image.open(im)) for im in orig_list])

        # Load segmented images and apply threshold
        seg_images = np.array([np.asarray(Image.open(im)) for im in seg_list])

        # Compute total pixels per image
        vessel_pixels = np.sum(seg_images > (thresh * 255.0), axis=(1, 2))
        pixel_totals[class_] = vessel_pixels

        # Sort the original images by total segmented vessel pixels
        order = np.argsort(vessel_pixels)

        sorted_orig = orig_images[order]
        sorted_seg = seg_images[order]

        sample = range(0, sorted_orig.shape[0], int(np.floor(sorted_orig.shape[0] / 10.0)))

        for i, idx in enumerate(sample):

            orig = Image.fromarray(sorted_orig[idx])
            seg = np.invert(Image.fromarray(sorted_seg[idx]))

            if i == 0:
                ax1[cidx, i].text(1., 1., class_, fontdict=font, verticalalignment='top')
                ax2[cidx, i].text(1., 1., class_, fontdict=font, verticalalignment='top')

            ax1[cidx, i].imshow(orig)
            ax1[cidx, i].axis('off')
            ax2[cidx, i].imshow(seg)
            ax2[cidx, i].axis('off')

        cidx += 1

    fig1.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    fig1.savefig(join(out_dir, 'orig_order.png'))

    fig2.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    fig2.savefig(join(out_dir, 'seg_order.png'))

    # Plot histogram
    fig, ax = plt.subplots()
    cols = ['r', 'g', 'b']

    for (class_, total_pixels), color in zip(pixel_totals.items(), cols):
        plt.hist(total_pixels, bins=10, normed=False, color=color, alpha=0.25, label=class_)

    plt.title("Total vessel pixels by class")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend(pixel_totals.keys())
    plt.savefig(join(out_dir, 'hist.png'))

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='in_dir', help='Folder of original images', required=True)
    parser.add_argument('-s', '--segmented', dest='seg_dir', help='Folder of segmented images', required=True)
    parser.add_argument('-o', '--output', dest='out_dir', help='Output folder', required=True)
    parser.add_argument('-t', '--threshold', dest='thresh', help='Threshold value', type=float, required=True)

    args = parser.parse_args()
    analyse_vessels(args.in_dir, args.seg_dir, args.out_dir, args.thresh)
