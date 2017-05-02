#!/usr/bin/env python

from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from os import listdir
from os.path import isdir, basename, abspath, dirname, splitext
from shutil import copy
import h5py

import addict
import pandas as pd
import yaml
from utils.common import make_sub_dir, find_images_by_class, find_images
from random import shuffle
from scipy.misc import imresize

from metadata import image_to_metadata
from methods import *
from segmentation.segment_unet import segment
from segmentation.mask_retina import *

METHODS = {'HN': normalize_channels, 'kaggle_BG': kaggle_BG, 'segment_vessels': segment,
           'unet_norm': unet_preproc,'morphology': binary_morph}
DEFAULT_CLASSES = ['No', 'Pre-Plus', 'Plus']


class Pipeline(object):

    def __init__(self, config):

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

                conf_dict = yaml.load(c)
                self.input_dir = abspath(join(dirname(config), conf_dict['input_dir']))
                self.out_dir = make_sub_dir(dirname(config), splitext(basename(config))[0])

                csv_file = abspath(join(dirname(config), conf_dict['csv_file']))
                self.labels = pd.DataFrame.from_csv(csv_file)
                self.reader = conf_dict['reader']

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
        im_files = find_images(join(self.input_dir, '*'))
        assert (len(im_files) > 0)

        # Split images into metadata
        imgs_split = [splitext(basename(x))[0].split('_') + [x] for x in im_files]
        imgs_split.sort(key=lambda x: x[1])  # sort by ID

        # Create DataFrame, indexed by ID
        df = pd.DataFrame(imgs_split, columns=['patient_id', 'id', 'session', 'view',
                                               'eye', 'class', 'full_path']).set_index('id')

        # Group by class/patient, and split into five
        all_splits = {}  # to keep track of all splits of the data

        for class_, c_group in df.groupby('class'):  # TODO user the specified reader, rather than filename

            p_groups = c_group.groupby('patient_id')  # group by patient

            # Create list of unique patients and randomly shuffle it
            all_patients = [pg for p_id, pg in p_groups]
            shuffle(all_patients)

            # Define split size to achieve 5 splits
            split_size = int(len(all_patients) * .2)
            all_splits[class_] = [pd.concat(all_patients[x:x + split_size]).sort_index() for x in
                                  range(0, len(all_patients) - split_size + 1, split_size)]

        # Split into training and testing
        train_test_splits = []

        split_range = set(range(0, 5))  # we want five training/testing sets
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
        for i in range(0, 5):  # for each split

            print "\n~~ Split {} ~~".format(i)

            train_split = train_test_splits[i]['train']  # get the training data
            test_split = train_test_splits[i]['test']  # get the testing data

            tr0 = train_split.shape[0]
            te0 = test_split.shape[0]

            # Calculate the total images in each set (it won't be exactly 80:20, but hopefully close)
            print "Train images %: {:.2f}".format(float(tr0) / (tr0 + te0) * 100)
            print "Test images %: {:.2f}\n".format(float(te0) / (tr0 + te0) * 100)

            # Check that the class distribution is maintained in each split
            print 'Class distribution - training:'
            print {class_: len(x) / float(tr0) for class_, x in train_split.groupby('class')}
            print 'Class distribution - testing:'
            print {class_: len(x) / float(te0) for class_, x in test_split.groupby('class')}

            split_dir = make_sub_dir(self.out_dir, 'split_{}'.format(i))
            train_dir = make_sub_dir(split_dir, 'training')
            test_dir = make_sub_dir(split_dir, 'testing')

            train_split.to_csv(join(split_dir, 'training.csv'))
            test_split.to_csv(join(split_dir, 'testing.csv'))

            for class_name in DEFAULT_CLASSES:  # need to make these in advance
                make_sub_dir(train_dir, class_name)
                make_sub_dir(test_dir, class_name)

            # Pre-process, augment and randomly sample the training set
            print "Preprocessing training data..."
            optimization_pool = Pool(self.processes)
            subprocess = partial(preprocess, args={'params': self, 'augment': True, 'out_dir': train_dir})
            train_imgs = list(train_split['full_path'])
            _ = optimization_pool.map(subprocess, train_imgs)

            self.generate_h5(find_images_by_class(train_dir), join(split_dir, 'train.h5'), random_sample=True)

            # Pre-process (but don't augment or randomly sample) the test set
            print "Preprocessing testing data..."
            optimization_pool = Pool(self.processes)
            subprocess = partial(preprocess, args={'params': self, 'augment': False, 'out_dir': test_dir})
            test_imgs = list(test_split['full_path'])
            _ = optimization_pool.map(subprocess, test_imgs)

            self.generate_h5(find_images_by_class(test_dir), join(split_dir, 'test.h5'), random_sample=False)

    def generate_h5(self, imgs, out_file, random_sample=True, classes=DEFAULT_CLASSES):

        train_class_sizes = {c: len(x) for c, x in imgs.items()}

        train_arr = []
        train_labels = []
        print '\n---'

        for cidx, class_ in enumerate(classes):

            removal_num = train_class_sizes[class_] - int(
                (float(min(train_class_sizes.values())) / float(train_class_sizes[class_])) * train_class_sizes[class_])

            if random_sample and removal_num > 0:
                random_train = self.choose_random(imgs[class_], removal_num)
                imgs[class_] = random_train

            for ti in imgs[class_]:
                train_arr.append(np.asarray(Image.open(ti)))
                train_labels.append(class_)

            print "{} ({}): {}".format(out_file, class_, len(imgs[class_]))

        # Save results
        train_data = np.transpose(np.asarray(train_arr), (0, 3, 2, 1))
        train_labels = np.asarray(train_labels)

        with h5py.File(out_file, "w") as f:
            f.create_dataset('data', data=train_data, dtype=train_data.dtype)
            f.create_dataset('labels', data=train_labels, dtype=train_labels.dtype)

    def choose_random(self, imgs, to_remove):
        return np.random.choice(imgs, len(imgs) - to_remove, replace=False)

    def preserve_originals(self, imgs, to_remove):

        # Sort the augmented images alphabetically and split into chunks (of augment_size)
        imgs = sorted(imgs)
        assert(len(imgs) % self.augment_size == 0)
        unique_chunks = [imgs[i:i+self.augment_size] for i in xrange(0, len(imgs), self.augment_size)]

        # Calculate how many images we need to remove in each chunk (some chunks likely smaller than others)
        total_proportion = float(to_remove) / float(len(imgs))
        remove_per_chunk = int(np.ceil(len(unique_chunks[0]) * total_proportion))

        # Loop through each chunk and sample the images needed
        subsampled = []
        for chunk in unique_chunks:

            # Randomly sample the chunk for image to keep
            sub_chunk = np.random.choice(chunk, self.augment_size - remove_per_chunk, replace=False)
            subsampled.extend(sub_chunk)

        return subsampled


def preprocess(im, args):

    print "Pre-processing '{}'".format(im)
    params, out_dir, augment = args['params'], args['out_dir'], args['augment']

    # Image metadata
    meta = image_to_metadata(im)
    im_ID = int(meta['imID'])

    # Get class and quality info
    row = params.labels.iloc[im_ID]
    reader = params.reader
    class_ = row[reader]
    stage = row['ROP_stage']
    quality = row['quality']

    # Skip images with invalid class, advanced ROP or insufficient quality
    if class_ not in DEFAULT_CLASSES or not quality or stage > 3:
        return False

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
    class_dir = join(out_dir, class_)

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

    return True

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)

    args = parser.parse_args()

    p = Pipeline(args.config)
    p.run()