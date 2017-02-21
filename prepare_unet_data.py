#!/usr/bin/env python

import os
import h5py
import numpy as np
from PIL import Image
import yaml
from os import listdir, mkdir
from os.path import join, isdir


class RetinalDataset(object):

    """
    This class is used to create HDF5 files in preparation for training a retina-unet.
    The code is largely based on that which is available here: https://github.com/orobix/retina-unet
    It has been modified to use a human-readable config file in YAML format.
    """

    def __init__(self, config_file):

        with open(config_file, 'rb') as y:
            self.config = yaml.load(y)

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
        for i, file_ in enumerate(listdir(imgs_dir)):

            print "Loading '{}'".format(file_)

            # Add the image to an ndarray
            img = Image.open(join(imgs_dir, file_))
            imgs_arr[i] = np.asarray(img)

            # Find the ground truth
            gt_file = file_[0:6] + "_manual1.gif"
            g_truth = Image.open(join(ground_truth_dir, gt_file))
            ground_truth_arr[i] = np.asarray(g_truth)

            # Find the mask
            ext = "_test_mask.gif"
            mask_file = file_[0:6] + ext
            b_mask = Image.open(join(mask_dir, mask_file))
            masks_arr[i] = np.asarray(b_mask)

        # Value assertions
        print "imgs max: {}".format(np.max(imgs_arr))
        print "imgs min: {}".format(np.min(imgs_arr))

        assert(np.max(ground_truth_arr) == 1 and np.max(masks_arr) == 1)
        assert(np.min(ground_truth_arr) == 0 and np.min(masks_arr) == 0)
        print "Ground truth and border masks are correctly within pixel value range 0 - 1 (black - white)"

        # Reshaping
        imgs_arr = np.transpose(imgs_arr,(0,3,1,2))
        assert(imgs_arr.shape == (self.n_imgs, self.channels, self.height, self.width))

        ground_truth_arr = np.reshape(ground_truth_arr,(self.n_imgs, 1, self.height, self.width))
        assert(ground_truth_arr.shape == (self.n_imgs, 1, self.height, self.width))

        masks_arr = np.reshape(masks_arr, (self.n_imgs, 1, self.height, self.width))
        assert(masks_arr.shape == (self.n_imgs, 1, self.height, self.width))
        
        # Write the data to disk
        type_ = 'train' if training else 'test'

        print "Saving dataset to '{}'".format(self.out_dir)
        write_hdf5(imgs_arr, join(self.out_dir, "200image_dataset_imgs_{}.hdf5".format(type_)))
        write_hdf5(ground_truth_arr, join(self.out_dir, "200image_dataset_groundTruth_{}.hdf5".format(type_)))
        write_hdf5(masks_arr, join(self.out_dir, "200image_dataset_borderMasks_{}.hdf5".format(type_)))

        return imgs_arr, ground_truth_arr, masks_arr


# #getting the training datasets
# print "getting training data"
# imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
# print "saving train datasets"
# write_hdf5(imgs_train, dataset_path + "200image_test_set_dataset_imgs_train.hdf5")
# write_hdf5(groundTruth_train, dataset_path + "200image_test_set_dataset_groundTruth_train.hdf5")
# write_hdf5(border_masks_train,dataset_path + "200image_test_set_dataset_borderMasks_train.hdf5")
#
# #getting the testing datasets
# imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
# print "saving test datasets"
# write_hdf5(imgs_test,dataset_path + "200image_test_set_dataset_imgs_test.hdf5")
# write_hdf5(groundTruth_test, dataset_path + "200image_test_set_dataset_groundTruth_test.hdf5")
# write_hdf5(border_masks_test,dataset_path + "200image_test_set_dataset_borderMasks_test.hdf5")


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


if __name__ == "__main__":

    import sys
    conf = sys.argv[1]

    rd = RetinalDataset(conf)

    print "Generating training data..."
    training_imgs, _training_ground_truth, training_masks = rd.create_dataset(training=True)

    print "Generating test data..."
    testing_imgs, testing_ground_truth, testing_masks = rd.create_dataset(training=False)
