#!/usr/bin/env python

from os.path import join
import pandas as pd
from scipy.misc import imresize
import cv2

from ..segmentation.segment_unet import SegmentUnet, segment
from ..utils.common import make_sub_dir
from qtim_ROP.retinaunet.lib.help_functions import *


def resize(im_dir, out_dir, csv_file, unet=None):

    csv_data = pd.DataFrame.from_csv(csv_file, index_col=None)

    if unet:
        unet = SegmentUnet(None, unet)

    for image, row in csv_data.iterrows():

        im_path = join(im_dir, row['image']) + '.jpeg'
        level = 'Healthy' if row['level'] < 2 else 'Diseased'

        im_arr = cv2.imread(im_path)[:, :, ::-1]
        resized_im = imresize(im_arr, (480, 640), interp='bicubic')

        if unet:
            resized_im = segment(resized_im, unet)

        level_dir = make_sub_dir(out_dir, level)
        visualize(resized_im, join(level_dir, row['image']))


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--in-dir', dest='in_dir', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)
    parser.add_argument('-u', '--unet', dest='unet', default=None)
    parser.add_argument('-c', '--csv-file', dest='labels', required=True)

    args = parser.parse_args()
    resize(args.in_dir, args.out_dir, args.labels, unet=args.unet)
