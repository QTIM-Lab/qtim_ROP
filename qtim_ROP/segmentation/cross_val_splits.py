import pandas as pd
from os import listdir
from os.path import join, isfile, splitext, basename, isdir
from ..utils.common import find_images, make_sub_dir, get_subdirs, write_hdf5
import scipy.io as sio
from PIL import Image
import numpy as np
# from utils.image import overlay_mask
from shutil import copy
from ..segmentation.train_unet import train_unet
import multiprocessing

UNET_SRC = '/root/eminas/James/ImageSets1-5/retina-unet'


def unet_cross_val(data_dir, out_dir, mapping, splits, unet_conf):

    # Load spreadsheet
    with pd.ExcelFile(mapping) as xls:
        df = pd.read_excel(xls, 'Sheet1').set_index('index')
        df['class'] = df['class'].map({'preplus': 'pre-plus', 'normal': 'normal', 'plus': 'plus'})

    img_dir = join(data_dir, 'images')
    seg_dir = join(data_dir, 'manual_segmentations')
    mask_dir = join(data_dir, 'masks')

    # Check whether all images exist
    check_images_exist(df, img_dir, seg_dir, mask_dir)

    # Now split into training and testing
    CVFile = sio.loadmat(splits)

    # # Combining Pre-Plus and Plus
    # trainPlusIndex = CVFile['trainPlusIndex'][0]
    # testPlusIndex = CVFile['testPlusIndex'][0]
    #
    # plus_dir = make_sub_dir(out_dir, 'trainTestPlus')
    # print "Generating splits for combined No and Pre-Plus"
    # generate_splits(trainPlusIndex, testPlusIndex, df, img_dir, mask_dir, seg_dir, plus_dir)

    # Combining No and Pre-Plus
    trainPrePIndex = CVFile['trainPrePIndex'][0]
    testPrePIndex = CVFile['testPrePIndex'][0]

    prep_dir = make_sub_dir(out_dir, 'trainTestPreP')
    print("Generating splits for combined Pre-Plus and Plus")
    generate_splits(trainPrePIndex, testPrePIndex, df, img_dir, mask_dir, seg_dir, prep_dir)

    # Train models
    train_and_test(prep_dir, unet_conf, processes=1)
    # train_and_test(plus_dir, unet_conf, processes=2)


def train_and_test(splits_dir, unet_conf, processes=2):

    conf_dicts = []

    for split in get_subdirs(splits_dir):

        # Train a model on this split's training data
        conf_file = join(split, 'configuration.txt')
        copy(unet_conf, conf_file)
        conf_dicts.append({'config_path': conf_file, 'unet_src': UNET_SRC})

    # Train several models in parallel
    pool = multiprocessing.Pool(processes)
    pool.map(train_unet, conf_dicts)


def generate_splits(trainIndex, testIndex, df, img_dir, mask_dir, seg_dir, out_dir):

    # Create training and test data for each of the five splits
    for i in range(trainIndex.shape[0]):

        split_dir = make_sub_dir(out_dir, 'split_{}'.format(i))
        images_dir = make_sub_dir(split_dir, 'images')

        if isdir(images_dir) and len(listdir(images_dir)) == 6:
            print("Data already processed for split #{}".format(i))
            continue

        # Training and testing indices for this split
        train_indices = np.squeeze(trainIndex[i])
        test_indices = np.squeeze(testIndex[i])

        # TRAIN
        train_images = [images_by_index(j, df, img_dir, col='filename') for j in train_indices]
        assert(len(train_images) == len(train_indices))
        train_masks = [images_by_index(j, df, mask_dir, col='filename') for j in train_indices]
        assert(len(train_masks) == len(train_indices))
        train_seg = [images_by_index(j, df, seg_dir, col='manual_segmentation') for j in train_indices]
        assert(len(train_seg) == len(train_indices))

        create_dataset(train_images, train_seg, train_masks, split_dir, 'train')

        # TEST
        test_images = [images_by_index(k, df, img_dir, col='filename') for k in test_indices]
        assert(len(test_images) == len(test_indices))
        test_masks = [images_by_index(k, df, mask_dir, col='filename') for k in test_indices]
        assert (len(test_masks) == len(test_indices))
        test_seg = [images_by_index(k, df, seg_dir, col='manual_segmentation') for k in test_indices]
        assert(len(test_seg) == len(test_indices))

        create_dataset(test_images, test_seg, test_masks, split_dir, 'test')


def create_dataset(img_paths, seg_paths, mask_paths, out_dir, name):

    n_imgs, height, width, channels = len(img_paths), 480, 640, 3

    # Initialise empty data arrays
    imgs_arr = np.empty((n_imgs, height, width, channels))
    ground_truth_arr = np.empty((n_imgs, height, width))
    masks_arr = np.empty((n_imgs, height, width))

    # Loop through all files in images directory
    for i, (img, seg, mask) in enumerate(zip(img_paths, seg_paths, mask_paths)):

        # print "Loading '{}'".format(img)

        img = Image.open(img)
        imgs_arr[i] = np.asarray(img)

        g_truth = Image.open(seg)
        ground_truth_arr[i] = (np.asarray(g_truth) > 0).astype(np.uint8)[:, :, 0] * 255

        b_mask = Image.open(mask)
        masks_arr[i] = np.asarray(b_mask)[:, :, 0]  # .astype(np.uint8)[:, :, 0] * 255

    # Value assertions
    print("imgs max: {}".format(np.max(imgs_arr)))
    print("imgs min: {}".format(np.min(imgs_arr)))

    assert (np.max(ground_truth_arr) == 255 and np.max(masks_arr) == 255)
    assert (np.min(ground_truth_arr) == 0 and np.min(masks_arr) == 0)
    print("Ground truth and border masks are correctly within pixel value range 0 - 255 (black - white)")

    # Reshaping
    imgs_arr = np.transpose(imgs_arr, (0, 3, 1, 2))
    assert (imgs_arr.shape == (n_imgs, channels, height, width))

    ground_truth_arr = np.reshape(ground_truth_arr, (n_imgs, 1, height, width))
    assert (ground_truth_arr.shape == (n_imgs, 1, height, width))

    masks_arr = np.reshape(masks_arr, (n_imgs, 1, height, width))
    assert (masks_arr.shape == (n_imgs, 1, height, width))

    # Write the data to disk
    print("Saving dataset to '{}'".format(out_dir))
    write_hdf5(imgs_arr, join(out_dir, "imgs_{}.hdf5".format(name)))
    write_hdf5(ground_truth_arr, join(out_dir, "ground_truth_{}.hdf5".format(name)))
    write_hdf5(masks_arr, join(out_dir, "masks_{}.hdf5".format(name)))

    return imgs_arr, ground_truth_arr, masks_arr


def check_overlay(imgs, masks, segs, out_dir, i=0):

    example_img, example_mask, example_seg = imgs[i], masks[i], segs[i]
    assert example_img.shape == example_mask.shape == example_seg.shape
    # overlay_mask(example_img * example_mask, example_seg[:, :, 0], join(out_dir, '{}.png'.format(i)))


def images_by_index(index, df, location, col='filename'):

    row = df.ix[index]

    img_path = join(location, row['class'], row[col])
    if not isfile(img_path):
        return None

    return img_path
    # arr = np.asarray(Image.open(img_path))
    # if col == 'manual_segmentation':
    #     return (arr > 0).astype(np.uint8)
    # else:
    #     return arr


def check_images_exist(df, img_dir, seg_dir, mask_dir):

    for index, row in df.iterrows():

        img_path = join(img_dir, row['class'], row['filename'])
        seg_path = join(seg_dir, row['class'], row['manual_segmentation'])
        mask_path = join(mask_dir, row['class'], row['filename'])

        try:
            assert (isfile(img_path))
            assert (isfile(seg_path))
            assert (isfile(mask_path))

        except AssertionError:
            print(img_path)
            print(seg_path)
            print(mask_path)
            exit()


def images_to_df(input_dir):

    # Get paths to all images
    im_files = find_images(join(input_dir, '*'))
    assert (len(im_files) > 0)

    # Split images into metadata
    imgs_split = [splitext(basename(x))[0].split('_') + [x] for x in im_files]
    imgs_split.sort(key=lambda x: x[1])  # sort by ID

    # Create DataFrame, indexed by ID
    return pd.DataFrame(imgs_split, columns=['patient_id', 'id', 'session', 'view', 'eye', 'class', 'full_path']).set_index('id')

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', '--images', dest='images', required=True, help="Folder of images, segs and masks")
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True, help="Output directory")
    parser.add_argument('-u', '--unet-conf', dest='unet_conf', required=True, help="U-Net configuration.txt")
    parser.add_argument('-m', '--mapping', dest='mapping', required=True, help="Excel file defining the order of the images")
    parser.add_argument('-sp', '--splits', dest='splits', required=True, help=".mat file containing the splits to be generated")

    args = parser.parse_args()
    unet_cross_val(args.images, args.out_dir, args.mapping, args.splits, args.unet_conf)
