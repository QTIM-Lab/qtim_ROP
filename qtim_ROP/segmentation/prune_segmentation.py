#!/usr/bin/env python

from os.path import join, basename
import cv2
from ..utils.common import find_images
from ..preprocessing.methods import binary_morph

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-dir', dest='in_dir', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)

    parser.add_argument('-t', '--thresh', dest='thresh', help="0 < thresh < 255", type=int, default=200)
    parser.add_argument('-s', '--smallest', dest='smallest', help="Smallest object size allowed", type=int, default=10)

    args = parser.parse_args()

    for im_path in find_images(args.in_dir):

        im_name = basename(im_path)
        img = cv2.imread(im_path)

        pruned = binary_morph(img, thresh=args.thresh, min_size=args.smallest)
        cv2.imwrite(join(args.out_dir, im_name), pruned)
