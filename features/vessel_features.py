from PIL import Image
import numpy as np
from glob import glob
from os.path import join, basename
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
from collections import defaultdict
from scipy.ndimage.morphology import distance_transform_edt
from features.tracing import VesselTree

from skimage.morphology import medial_axis, skeletonize
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
import cv2

from geom import *
from utils.common import make_sub_dir

CLASS_LABELS = {'normal': 0, 'pre-plus': 1, 'plus': 2}


def vessel_features(orig_dir, seg_dir, out_dir, csv_file):

    csv_data = pd.DataFrame.from_csv(csv_file)

    orig_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(orig_dir, '*.*')))]
    seg_images = [np.asarray(Image.open(x)) for x in sorted(glob(join(seg_dir, '*.png')))]
    prob = .5

    vessel_dir = make_sub_dir(out_dir, 'mask')
    skel_dir = make_sub_dir(out_dir, 'skel')
    tracing_dir = make_sub_dir(out_dir, 'tracing')

    features = defaultdict(list)
    y = []

    for i, (orig, seg) in enumerate(zip(orig_images, seg_images)):

        # Extract row
        csv_row = csv_data.iloc[i]
        class_ = csv_row['class_name']
        od_center = csv_row['optic_disk_x'], csv_row['optic_disk_y']
        img_name = '{}_{}.png'.format(i, class_)
        y.append(CLASS_LABELS[class_])

        print "Processing '{}'".format(img_name)

        # Binarize and overlay
        vessel_mask = (seg > (255 * prob)).astype(np.uint8)
        overlay_mask(orig, vessel_mask, join(vessel_dir, img_name))

        # Extract medial axis
        skel = skeletonize(vessel_mask)

        # Remove small isolated segments
        labelled = label(skel)
        cleaned_skel = remove_small_objects(labelled, min_size=50)
        cleaned_skel = cleaned_skel > 0
        overlay_mask(orig, cleaned_skel.astype(np.uint8) * 255, join(skel_dir, img_name))

        # Create masked skeleton
        masked_skel = mask_od_vessels(cleaned_skel, od_center)

        # Compute vessel tree
        tree = VesselTree(rgb2gray(orig), masked_skel, tracing_dir, img_name)
        tree.run()

        # features = tree.get_features()

        # # Hough transform + line segments
        # lines = probabilistic_hough_line(cleaned_skel, line_length=2, line_gap=1)
        # plot_lines(orig, lines, join(hough_dir, img_name))
        #
        # # Distance transform
        # edt = distance_transform_edt(vessel_mask)
        #
        # features['total'].append(len(lines))
        # features['angles'].append(pairwise_angles(lines))
        # features['lengths'].append(line_lengths(lines))
        # features['thickness'].append(vessel_thickness(edt, cleaned_skel))


def normalize(img):

    img_min, img_max = np.min(img), np.max(img)
    img_norm = (img - img_min) / (img_max - img_min)
    return img_norm.astype(np.uint8) * 255


def plot_lines(img, lines, out, lw=2.0):

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    for line in lines:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), alpha=0.7, linewidth=lw)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()

if __name__ == '__main__':

    import sys

    root_dir = sys.argv[1]
    orig_dir = join(root_dir, 'images')
    seg_dir = join(root_dir, 'vessels')
    csv_file = join(root_dir, 'mapping.csv')
    out_dir = join(root_dir, 'analysis')

    vessel_features(orig_dir, seg_dir, out_dir, csv_file)