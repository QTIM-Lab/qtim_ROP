#!/usr/bin/env python

from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from os.path import isdir, basename, abspath, dirname, splitext, join
import h5py

import addict
import pandas as pd
import yaml
from random import shuffle, seed
from PIL import Image

import cv2
from scipy.misc import imresize
from os.path import join
import numpy as np
from PIL import Image

from ..utils.common import make_sub_dir, find_images_by_class, find_images
from ..utils.metadata import image_to_metadata
from .methods import *


# Set the random seed
seed(101)

METHODS = {'HN': normalize_channels, 'kaggle_BG': kaggle_BG,  # 'segment_vessels': segment,
           'unet_norm': unet_preproc, 'morphology': binary_morph}
DEFAULT_CLASSES = ['No', 'Pre-Plus', 'Plus']


class Pipeline(object):

    def __init__(self, config, n_folds=5, out_dir=None, exclusions=None):

        self.out_dir = out_dir
        self.n_folds = int(n_folds)
        self.exclusions = exclusions
        self._parse_config(config)

        # Define preprocessor
        if self.pipeline.preprocessing:
            method = self.pipeline.preprocessing['method']
            self.preprocessor = METHODS.get(method, None)
            self.p_args = self.pipeline.preprocessing['args']
        else:
            self.preprocessor = None

        # Number of processes
        self.processes = int(cpu_count())

    def _parse_config(self, config):

        try:
            with open(config, 'rb') as c:

                # Load config file and parse
                conf_dict = yaml.load(c)
                self.input_dir = abspath(join(dirname(config), conf_dict['input_dir']))

                if not self.out_dir:
                    self.out_dir = make_sub_dir(dirname(config), splitext(basename(config))[0])

                csv_file = abspath(join(dirname(config), conf_dict['csv_file']))
                self.label_data = pd.DataFrame.from_csv(csv_file)
                self.label = conf_dict.get('reader')

                if not isdir(self.input_dir):
                    print "Input {} is not a directory!".format(self.input_dir)
                    exit()

                # Extract pipeline parameters or set defaults
                self.pipeline = addict.Dict(conf_dict['pipeline'])
                self.augment_size = self.pipeline.augmentation['size']
                self.resize = self.pipeline['resize']
                self.crop = self.pipeline.get('crop', None)

                if self.crop:
                    self.crop_width = (self.resize['width'] - self.crop['width']) / 2
                    self.crop_height = (self.resize['height'] - self.crop['height']) / 2

        except KeyError as e:
            print "Missing config entry {}".format(e)
            exit()

    def run(self):

        # Get paths to all images
        # im_files = find_images(join(self.input_dir, '*'))
        im_files = find_images(join(self.input_dir))
        assert (len(im_files) > 0)

        # Split images into metadata
        imgs_split = [splitext(basename(x))[0].split('_') + [x] for x in im_files]
        imgs_split.sort(key=lambda x: x[1])  # sort by ID

        # Create DataFrame of all available images
        img_data = pd.DataFrame(imgs_split, columns=['patient_id', 'id', 'session', 'view',
                                               'eye', 'class', 'full_path']).set_index('id')
        img_data.index = img_data.index.map(int)  # convert from string to integer

        # Add labels
        img_data = img_data.join(self.label_data)

        # Consolidate stage labels, or remove late stage entries
        if self.label == 'ROP_stage':

            # Binary classification: 2 or better vs. 3 or worse
            img_data.loc[img_data['ROP_stage'] <= 2, 'ROP_stage'] = 0
            img_data.loc[img_data['ROP_stage'] > 2, 'ROP_stage'] = 1

            # df['ROP_stage'][df['ROP_stage'] <= 2] = 0
            # df['ROP_stage'][df['ROP_stage'] > 2] = 1
            assert (np.max(img_data['ROP_stage']) == 1)
        else:
            # Remove stage 4 and 5 ROP cases
            img_data = img_data[img_data.ROP_stage < 4]
            img_data = img_data[img_data.quality]

        self.label_data = img_data

        # Add a column with the names of the original images
        # orig_names = [self.label_data.iloc[i]['imageName'] for i in img_data.index.values if i in self.label_data.index]
        # img_data['original'] = orig_names

        # Check that the filename IDs match the data frame IDs
        assert(all(int(basename(x['full_path']).split('_')[1]) == int(index) for index, x in img_data.iterrows()))

        if self.n_folds == 1:

            exclude_patients = pd.Series.from_csv(self.exclusions).values
            img_data = img_data[~img_data['patient_id'].isin(exclude_patients)]
            img_data.to_csv(join(self.out_dir, 'training.csv'))
            self.generate_dataset(self.out_dir, mode='training')
            quit()

        # Group by class/patient, and split into folds
        all_splits = {}  # to keep track of all splits of the data

        # Split by class (reader)
        for class_, c_group in img_data.groupby(self.label):

            p_groups = c_group.groupby('patient_id')  # group by patient

            # Create list of unique patients and randomly shuffle it
            all_patients = [pg for p_id, pg in p_groups]
            shuffle(all_patients)

            # Define split size to achieve n splits
            split_size = int(len(all_patients) * (1. / self.n_folds))
            all_splits[class_] = [pd.concat(all_patients[x:x + split_size]).sort_index() for x in
                                  range(0, len(all_patients) - split_size + 1, split_size)]

        # Split into training and testing
        train_test_splits = []

        split_range = set(range(0, self.n_folds))  # we want five training/testing sets
        for test_index in split_range:

            train_split = []  # to store training
            test_split = []  # to store testing

            for class_ in all_splits.keys():  # for each class

                class_splits = all_splits[class_]  # grab all splits for this class
                train_indices = list(split_range - set([test_index]))  # chunks of patients to train on
                train_split.extend([all_splits[class_][t] for t in train_indices])
                test_split.append(class_splits[test_index])  # test on remaining patients

            # Concatenate the training / testing for each class
            train_test_splits.append({'train': pd.concat(train_split), 'test': pd.concat(test_split)})

        # For each split
        for i in split_range:  # for each split

            print "\n~~ Split {} ~~".format(i)

            train_split = train_test_splits[i]['train']  # get the training data
            test_split = train_test_splits[i]['test']  # get the testing data

            # There is a chance that data from the same patient are split in both sets
            patient_intersection = set.intersection(set(train_split['patient_id'].values), set(test_split['patient_id'].values))

            for pat in patient_intersection:

                train_patients = train_split.loc[train_split['patient_id'] == pat]
                test_patients = test_split.loc[test_split['patient_id'] == pat]

                move_to_train = np.argmax([test_patients.shape[0], train_patients.shape[0]])
                if move_to_train:
                    train_split = train_split.append(test_patients)  # copy from test to train
                    test_split = test_split[test_split.patient_id != pat]  # remove from test
                else:
                    test_split = test_split.append(train_patients)  # copy from train to test
                    train_split = train_split[train_split.patient_id != pat]  # remove from train

            # Re-check that there is no patient overlap
            patient_intersection = set.intersection(set(train_split['patient_id'].values), set(test_split['patient_id'].values))
            assert(len(patient_intersection) == 0)

            # Confirms that the testing and training are split properly - same images don't appear in both
            assert(all(x not in train_split.index.values for x in test_split.index.values))

            tr0 = train_split.shape[0]
            te0 = test_split.shape[0]

            # Calculate the total images in each set (it won't be exact but hopefully close)
            print "Train images %: {:.2f}".format(float(tr0) / (tr0 + te0) * 100)
            print "Test images %: {:.2f}\n".format(float(te0) / (tr0 + te0) * 100)

            # Check that the class distribution is maintained in each split
            print 'Class distribution - training:'
            print {class_: len(x) / float(tr0) for class_, x in train_split.groupby(self.label)}
            print 'Class distribution - testing:'
            print {class_: len(x) / float(te0) for class_, x in test_split.groupby(self.label)}

            split_dir = make_sub_dir(self.out_dir, 'split_{}'.format(i))
            train_split.to_csv(join(split_dir, 'training.csv'))
            test_split.to_csv(join(split_dir, 'testing.csv'))

            self.generate_dataset(split_dir, mode='training')
            self.generate_dataset(split_dir, mode='testing')

    def generate_dataset(self, split_dir, mode='training'):

        if mode not in ['training', 'testing']:
            raise ValueError("Mode must be 'training' or 'testing'")

        do_augment = mode == 'training'  # we only want to augment the training data
        split_df = pd.DataFrame.from_csv(join(split_dir, '{}.csv'.format(mode)))  # load splits
        data_dir = make_sub_dir(split_dir, mode)  # output directory for images

        # Make directories for each class of images in advance
        classes = [str(l) for l in split_df[self.label].unique()]
        for class_name in classes:
            make_sub_dir(data_dir, str(class_name))

        # Pre-process, augment and randomly sample the training set
        print "Preprocessing {} data...".format(mode)

        if len(find_images(join(data_dir, '*'))) == 0:
            pool = Pool(self.processes)
            subprocess = partial(do_preprocess, args={'params': self, 'augment': do_augment, 'out_dir': data_dir})
            img_list = list(split_df['full_path'])
            _ = pool.map(subprocess, img_list)

        self.generate_h5(find_images_by_class(data_dir, classes=classes), join(split_dir, '{}.h5'.format(mode)), split_df,
                         random_sample=True, classes=classes)

    def generate_h5(self, imgs, out_file, df, random_sample=True, classes=DEFAULT_CLASSES):

        class_sizes = {c: len(x) for c, x in imgs.items()}

        img_arr = []
        img_labels = []
        original_images = []

        for cidx, class_ in enumerate(classes):

            removal_num = class_sizes[class_] - int(
                (float(min(class_sizes.values())) / float(class_sizes[class_])) * class_sizes[class_])

            if random_sample and removal_num > 0:
                random_train = self.choose_random(imgs[class_], removal_num)
                imgs[class_] = random_train

            for img_path in imgs[class_]:

                try:
                    id_ = int(basename(img_path).split('_')[1])
                    original_image = df.loc[id_]['imageName']
                except KeyError:
                    raise

                assert(all(original_image.split('_')[j] == basename(img_path).split('_')[j] for j in range(0, 5)))

                img_arr.append(np.asarray(Image.open(img_path)))
                img_labels.append(class_)
                original_images.append(original_image)

            print "{} ({}): {}".format(out_file, class_, len(imgs[class_]))

        # Save results
        train_data = np.transpose(np.asarray(img_arr), (0, 3, 2, 1))
        img_labels = np.asarray(img_labels)
        original_images = np.asarray(original_images)

        with h5py.File(out_file, "w") as f:
            f.create_dataset('data', data=train_data, dtype=train_data.dtype)
            f.create_dataset('labels', data=img_labels, dtype=img_labels.dtype)
            f.create_dataset('original_files', data=original_images, dtype=original_images.dtype)

    def choose_random(self, imgs, to_remove):
        return np.random.choice(imgs, len(imgs) - to_remove, replace=False)

    # def preserve_originals(self, imgs, to_remove):
    #
    #     # Sort the augmented images alphabetically and split into chunks (of augment_size)
    #     imgs = sorted(imgs)
    #     assert(len(imgs) % self.augment_size == 0)
    #     unique_chunks = [imgs[i:i+self.augment_size] for i in xrange(0, len(imgs), self.augment_size)]
    #
    #     # Calculate how many images we need to remove in each chunk (some chunks likely smaller than others)
    #     total_proportion = float(to_remove) / float(len(imgs))
    #     remove_per_chunk = int(np.ceil(len(unique_chunks[0]) * total_proportion))
    #
    #     # Loop through each chunk and sample the images needed
    #     subsampled = []
    #     for chunk in unique_chunks:
    #
    #         # Randomly sample the chunk for image to keep
    #         sub_chunk = np.random.choice(chunk, self.augment_size - remove_per_chunk, replace=False)
    #         subsampled.extend(sub_chunk)
    #
    #     return subsampled


def do_preprocess(im, args):

    # print "Pre-processing '{}'".format(im)
    params, out_dir, augment = args['params'], args['out_dir'], args['augment']

    # Image metadata
    meta = image_to_metadata(im)
    im_ID = int(meta['imID'])

    # Get class and quality info
    row = params.label_data.loc[im_ID]
    reader = params.label  # TODO make duplicates for all available readers if this is equal to 'all'
    class_ = row[reader]

    # Resize, preprocess and augment
    try:
        im_arr = cv2.imread(im)[:, :, ::-1]
    except TypeError:
        print "Error loading '{}'".format(im)
        return False

    im_arr = im_arr[:,:,:3]

    # Resize and preprocess
    interp = params.resize.get('interp', 'bilinear')
    resized_im = imresize(im_arr, (params.resize['height'], params.resize['width']), interp=interp)

    if params.preprocessor:
        preprocessed_im = params.preprocessor(resized_im, *params.p_args)
    else:
        preprocessed_im = resized_im

    # Crop
    if params.crop is not None:
        preprocessed_im = preprocessed_im[params.crop_height:-params.crop_height, params.crop_width:-params.crop_width]

    # Augment/save
    class_dir = join(out_dir, str(class_))

    if augment:

        img = preprocessed_im
        flip_names = ['noflip', 'flip']
        rotate_names = ['0', '90', '180', '270']

        for flip in [0, 1]:
            for rotate in [0, 1, 2, 3]:
                new_img = np.copy(img)
                if rotate > 0:
                    for i in xrange(rotate):
                        new_img = np.rot90(new_img)
                if flip == 1:
                    new_img = np.fliplr(new_img)
                im = Image.fromarray(new_img)

                out_name = '{}_{}_{}.png'.format(meta['prefix'], flip_names[flip], rotate_names[rotate])
                out_path = join(class_dir, out_name)
                im.save(out_path)

    else:
        Image.fromarray(preprocessed_im).save(join(class_dir, meta['prefix'] + '.png'))

    return im
