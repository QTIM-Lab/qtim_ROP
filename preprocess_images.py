#!/usr/bin/env python

import cv2, numpy
from glob import glob
from os.path import join, isdir, basename
from methods import *


METHODS = {'bg': kaggle_BG}
CLASSES = ['No', 'Pre-Plus', 'Plus']
SCALE = 256


def preprocessing(in_path, out_path, method='bg', m_args=None):

    if not isdir(out_path):
        print "{} is not a directory. Please create this directory first!".format(out_path)
        exit()

    # Get the desired method function
    func = METHODS.get(method, 'bg')

    for c in CLASSES:

        im_list = find_images(join(in_path, c))
        assert(len(im_list) > 0)

        for im in im_list:
            try:

                proc_im = func(im, *m_args)
                base_name = basename(im)
                new_file = join(out_path, c, base_name)

                print "Writing preprocessed image to '{}'".format(new_file)
                cv2.imwrite(new_file, proc_im)

            except:
                print(im)


def find_images(im_path):

    files = []
    for ext in ('*.bmp', '*.BMP', '*.png', '*.jpg', '*.tif'):
        files.extend(glob(join(im_path, ext)))

    return sorted(files)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--input-path', dest='input_path', help="Input images to preprocess", required=True)
    parser.add_argument('-o', '--output-path', dest='output_path', help="Path to output processed images", required=True)
    parser.add_argument('-m', '--method', dest='method', help="Preprocessing method", choices=('bg',), default='bg')
    parser.add_argument('-a', '--args', dest='m_args', help="Preprocessing arguments", nargs='+', default=(128, .95))

    args = parser.parse_args()
    preprocessing(args.input_path, args.output_path, method=args.method, m_args=args.m_args)
