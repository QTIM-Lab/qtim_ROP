#!/usr/bin/env python

from PIL import Image
from os import listdir, mkdir, chdir
from os.path import join, isdir, abspath, dirname, basename

import numpy as np
import yaml
from ..utils.common import write_hdf5
from scipy.misc import imresize
from .mask_retina import create_mask


class RetinalDataset(object):

    """
    This class is used to create HDF5 files in preparation for training a retina-unet.
    The code is largely based on that which is available here: https://github.com/orobix/retina-unet
    It has been modified to use a human-readable config file in YAML format.
    """

    def __init__(self, config_file):

        with open(config_file, 'rb') as y:
            self.config = yaml.load(y)

        chdir(dirname(abspath(config_file)))

        self.dataset_path = self.config['dataset_path']
        assert(isdir(self.dataset_path))

        self.n_imgs = self.config['n_imgs']
        self.height = self.config['height']
        self.width = self.config['width']
        self.channels = 3

        # Directory of the training and testing data
        self.training = join(self.dataset_path, 'training')
        self.testing = join(self.dataset_path, 'test')
        self.out_dir = self.config['out_dir']

        if not isdir(self.out_dir):
            mkdir(self.out_dir)

    def create_dataset(self, training=True):

        # Root of the data
        data_dir = self.training if training else self.testing

        # Subdirectories for images, ground truth and masks
        imgs_dir = join(data_dir, 'images')
        ground_truth_dir = join(data_dir, '1st_manual')
        mask_dir = join(data_dir, 'mask')

        # Initialise empty data arrays
        imgs_arr = np.empty((self.n_imgs, self.height, self.width, self.channels))
        ground_truth_arr = np.empty((self.n_imgs, self.height, self.width))
        masks_arr = np.empty((self.n_imgs, self.height, self.width))

        # Loop through all files in images directory
        for i, file_ in enumerate(sorted(listdir(imgs_dir))):

            print("Loading '{}'".format(file_))

            # Add the image to an ndarray
            img = Image.open(join(imgs_dir, file_))
            imgs_arr[i] = np.asarray(img)

            # Find the ground truth
            gt_file = file_[0:6] + "_manual1.gif"
            g_truth = Image.open(join(ground_truth_dir, gt_file))
            ground_truth_arr[i] = np.asarray(g_truth).astype(np.uint8) * 255

            # Find the mask
            ext = "_test_mask.gif"
            mask_file = file_[0:6] + ext
            b_mask = Image.open(join(mask_dir, mask_file))
            masks_arr[i] = np.asarray(b_mask).astype(np.uint8) * 255

        # Value assertions
        print("imgs max: {}".format(np.max(imgs_arr)))
        print("imgs min: {}".format(np.min(imgs_arr)))

        assert(np.max(ground_truth_arr) == 255 and np.max(masks_arr) == 255)
        assert(np.min(ground_truth_arr) == 0 and np.min(masks_arr) == 0)
        print("Ground truth and border masks are correctly within pixel value range 0 - 255 (black - white)")

        # Reshaping
        imgs_arr = np.transpose(imgs_arr,(0,3,1,2))
        assert(imgs_arr.shape == (self.n_imgs, self.channels, self.height, self.width))

        ground_truth_arr = np.reshape(ground_truth_arr,(self.n_imgs, 1, self.height, self.width))
        assert(ground_truth_arr.shape == (self.n_imgs, 1, self.height, self.width))

        masks_arr = np.reshape(masks_arr, (self.n_imgs, 1, self.height, self.width))
        assert(masks_arr.shape == (self.n_imgs, 1, self.height, self.width))

        # Write the data to disk
        type_ = 'train' if training else 'test'

        print("Saving dataset to '{}'".format(self.out_dir))
        write_hdf5(imgs_arr, join(self.out_dir, "image_dataset_imgs_{}.hdf5".format(type_)))
        write_hdf5(ground_truth_arr, join(self.out_dir, "image_dataset_groundTruth_{}.hdf5".format(type_)))
        write_hdf5(masks_arr, join(self.out_dir, "image_dataset_borderMasks_{}.hdf5".format(type_)))

        return imgs_arr, ground_truth_arr, masks_arr


def imgs_to_unet_array(img_list, target_shape=(480, 640, 3), erode=10):

    height, width, channels = target_shape

    imgs_arr = []  # np.empty((n_imgs, height, width, channels))
    masks_arr = []  # np.empty((n_imgs, height, width, 1), dtype=np.bool)
    skipped = []

    for i, im_path in enumerate(img_list):

        try:
            img = np.asarray(Image.open(im_path))
        except IOError:
            print("Error loading image '{}' - skipping".format(im_path))
            skipped.append(im_path)
            continue

        if not img.shape:
            print("'{}' has invalid image shape - skipping".format(im_path))
            skipped.append(im_path)
            continue

        img = img[:, :, :3]  # in case there's an alpha channel

        if img.shape[:-1] != target_shape[:-1]:
            img = imresize(img, (height, width), interp='bicubic')

        print('{}: {}'.format(basename(im_path), img.shape))
        imgs_arr.append(img)

        mask = create_mask(img, erode=erode)
        masks_arr.append(np.expand_dims(mask, 2))

    imgs_arr = np.stack(imgs_arr, axis=0)
    masks_arr = np.stack(masks_arr, axis=0)

    # imgs_arr = np.transpose(imgs_arr, (0, 3, 1, 2))
    # masks_arr = np.transpose(masks_arr, (0, 3, 1, 2))

    return imgs_arr, masks_arr, skipped


if __name__ == "__main__":

    import sys
    conf = sys.argv[1]

    rd = RetinalDataset(conf)

    print("Generating training data...")
    training_imgs, _training_ground_truth, training_masks = rd.create_dataset(training=True)

    print("Generating test data...")
    testing_imgs, testing_ground_truth, testing_masks = rd.create_dataset(training=False)
