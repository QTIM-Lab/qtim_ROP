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
from random import shuffle, seed
from scipy.misc import imresize

from metadata import image_to_metadata
from methods import *
from segmentation.segment_unet import segment
from segmentation.mask_retina import *

# Set the random seed
seed(101)

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
        # im_files = find_images(join(self.input_dir, '*'))
        im_files = find_images(join(self.input_dir))
        assert (len(im_files) > 0)

        # Split images into metadata
        imgs_split = [splitext(basename(x))[0].split('_') + [x] for x in im_files]
        imgs_split.sort(key=lambda x: x[1])  # sort by ID

        # Create DataFrame, indexed by ID
        df = pd.DataFrame(imgs_split, columns=['patient_id', 'id', 'session', 'view',
                                               'eye', 'class', 'full_path']).set_index('id')

        # Add a column with the names of the original images
        orig_names = [self.labels.iloc[int(i)]['imageName'] for i in df.index.values]
        df['original'] = orig_names

        assert(all(int(x['original'].split('_')[1]) == int(index) for index, x in df.iterrows()))

        # Group by class/patient, and split into five
        all_splits = {}  # to keep track of all splits of the data

        for class_, c_group in df.groupby('class'):  # TODO allow specific reader labels, rather than RSD labels

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

            if len(find_images(join(train_dir, '*'))) == 0:
                pool = Pool(self.processes)
                subprocess = partial(preprocess, args={'params': self, 'augment': True, 'out_dir': train_dir})
                train_imgs = list(train_split['full_path'])
                _ = pool.map(subprocess, train_imgs)

            self.generate_h5(find_images_by_class(train_dir), join(split_dir, 'train.h5'), train_split, random_sample=True)

            # Pre-process (but don't augment or randomly sample) the test set
            print "Preprocessing testing data..."
            if len(find_images(join(test_dir, '*'))) == 0:

                pool = Pool(self.processes)
                subprocess = partial(preprocess, args={'params': self, 'augment': False, 'out_dir': test_dir})
                test_imgs = list(test_split['full_path'])
                _ = pool.map(subprocess, test_imgs)

            self.generate_h5(find_images_by_class(test_dir), join(split_dir, 'test.h5'), test_split, random_sample=False)

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
                    id = basename(img_path).split('_')[1]
                    original_image = df.loc[id]['original']
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


def preprocess(im, args):

    # print "Pre-processing '{}'".format(im)
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

    return im

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)

    args = parser.parse_args()

    p = Pipeline(args.config)
    p.run()