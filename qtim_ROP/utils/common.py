from glob import glob
from os import walk, mkdir, name as os_name
from os.path import *
from shutil import copytree
import logging
import sys
from PIL import Image
import numpy as np
import pandas as pd
import h5py
import yaml

DEFAULT = ['No', 'Pre-Plus', 'Plus']
EXTENSIONS = ['*.bmp', '*png', '*.jpg', '*.jpeg', '*.tif', '*tiff', '*.gif']
if os_name != 'nt':
    EXTENSIONS.extend([ext.upper() for ext in EXTENSIONS])


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
    for ext in EXTENSIONS:
        files.extend(glob(join(im_path, ext)))

    return sorted(files)


def find_images_by_class(im_path, classes=None, numeric=False):

    if classes is None:
        classes = DEFAULT

    images = {}
    for i, class_ in enumerate(classes):

        key = i if numeric else class_
        images[key] = find_images(join(im_path, class_))

    return images


def find_images_recursive(src_dir):

    matches = []
    for root, dirnames, filenames in walk(src_dir):

        for filename in filenames:
            if '*' + splitext(filename)[1] in EXTENSIONS:
                matches.append(join(root, filename))
    return matches


def get_subdirs(root_dir):

    return [x for x in glob(join(root_dir, '*')) if isdir(x)]


def write_hdf5(arr, outfile, name="image"):
    with h5py.File(outfile, "w") as f:
        f.create_dataset(name, data=arr, dtype=arr.dtype)


def imgs_and_labels_to_hdf5(data, labels, out_path):
    with h5py.File(join(out_path), "w") as f:
        f.create_dataset('data', data=data, dtype=data.dtype)
        f.create_dataset('labels', data=labels, dtype=labels.dtype)


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

    return logger


def dict_to_csv(my_dict, my_csv):

    pd.DataFrame(my_dict).to_csv(my_csv)


def csv_to_dict(my_csv):

    return pd.read_csv(my_csv).to_dict(orient='list')


def series_to_plot_dict(series, key, value):

    sorted_list = [{key: k, value: v} for k, v in list(series.to_dict().items())]
    return pd.DataFrame(data=sorted_list)


def dict_reverse(my_dict):

    return {v: k for k, v in list(my_dict.items())}