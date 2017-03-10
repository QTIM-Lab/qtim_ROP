from glob import glob
from os import mkdir
from os.path import join, isdir, isfile
from shutil import copytree
import logging
import sys
from collections import defaultdict

import pandas as pd
import h5py
from PIL import Image
import numpy as np
import yaml
from mask_retina import create_mask

CLASSES = ['No', 'Pre-Plus', 'Plus']


def make_sub_dir(dir_, sub, tree=None):

    sub_dir = join(dir_, sub)
    if not isdir(sub_dir):

        if tree:
            copytree(tree, sub_dir, ignore=ignore_files)
        else:
            mkdir(sub_dir)

    return sub_dir


def ignore_files(dir, files):
    return [f for f in files if isfile(join(dir, f))]


def find_images(im_path):

    files = []
    for ext in ['*.bmp', '*.BMP', '*.png', '*.jpg', '*.tif']:
        files.extend(glob(join(im_path, ext)))

    return sorted(files)


def find_images_by_class(im_path):

    images = {}
    for class_ in CLASSES:
        images[class_] = find_images(join(im_path, class_))

    return images

def imgs_to_unet_array(img_list):

    n_imgs = len(img_list)
    test_im = np.asarray(Image.open(img_list[0]))
    width, height, channels = test_im.shape

    imgs_arr = np.empty((n_imgs, width, height, channels))
    masks_arr = np.empty((n_imgs, width, height, 1), dtype=np.bool)

    for i, im_path in enumerate(img_list):

        img = np.asarray(Image.open(im_path))
        imgs_arr[i] = img

        mask = create_mask(img, erode=5)
        masks_arr[i] = np.expand_dims(mask, 2)

    imgs_arr = np.transpose(imgs_arr, (0, 3, 1, 2))
    masks_arr = np.transpose(masks_arr, (0, 3, 1, 2))

    return imgs_arr, masks_arr


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def parse_yaml(conf_file):

    with open(conf_file, 'r') as f:
        return yaml.load(f)


def setup_log(log_file, to_file=False):

    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if to_file:
        l_open = open(log_file, 'a')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sys.stdout, sys.stderr = l_open, l_open
    else:
        sout = logging.StreamHandler(sys.stdout)
        sout.setFormatter(fmt)
        logger.addHandler(sout)


def dict_to_csv(my_dict, my_csv):

    pd.DataFrame(my_dict).to_csv(my_csv)


def csv_to_dict(my_csv):

    return pd.read_csv(my_csv).to_dict(orient='list')
