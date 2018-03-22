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
from ..utils.common import find_images, make_sub_dir
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize

from ..utils.metadata import image_to_metadata
from .methods import *
from ..segmentation.segment_unet import segment, SegmentUnet
from ..segmentation.mask_retina import *

METHODS = {'HN': normalize_channels, 'kaggle_BG': kaggle_BG, 'segment_vessels': segment,
           'unet_norm': unet_preproc,'morphology': binary_morph}
DEFAULT_CLASSES = ['No', 'Pre-Plus', 'Plus']


class Pipeline(object):

    def __init__(self, config, exclusions=None):

        self._parse_config(config)
        self.exclusions = exclusions

        # Calculate class distribution and train/val split
        # self.class_distribution = {c: len(listdir(join(self.input_dir, c))) for c in DEFAULT_CLASSES}
        # largest_class = max(self.class_distribution, key=lambda k: self.class_distribution[k])
        # class_size = self.class_distribution[largest_class]

        # Make directories for intermediate steps
        self.augment_dir = make_sub_dir(self.out_dir, 'augmented', tree=self.input_dir)
        self.train_dir = make_sub_dir(self.out_dir, 'training', tree=self.input_dir)
        self.val_dir = make_sub_dir(self.out_dir, 'validation', tree=self.input_dir)

        # Define preprocessor
        if self.pipeline.preprocessing:

            method = self.pipeline.preprocessing['method']
            self.preprocessor = METHODS.get(method, None)
            p_args = self.pipeline.preprocessing['args']

            if method == 'segment_vessels':  # pre-instantiate retina-unet
                self.p_args = [SegmentUnet(None, *p_args)]
            else:
                self.p_args = p_args

        else:
            self.preprocessor = None

        # Create augmenter
        if self.pipeline.augmentation['method'] == 'keras':
            self.augmenter = ImageDataGenerator(
                width_shift_range=float(self.resize['width']) * 1e-4,
                height_shift_range=float(self.resize['height']) * 1e-4,
                zoom_range=0.05, horizontal_flip=True, vertical_flip=True, fill_mode='constant')
        else:
            self.augmenter = None

        # Number of processes
        self.processes = int(cpu_count())

    def _parse_config(self, config):

        try:
            with open(config, 'rb') as c:

                conf_dict = yaml.load(c)
                self.input_dir = abspath(join(dirname(config), conf_dict['input_dir']))
                self.out_dir = make_sub_dir(dirname(config), splitext(basename(config))[0])

                self.class_merge = conf_dict.get('class_merge', None)

                csv_file = abspath(join(dirname(config), conf_dict['csv_file']))
                self.labels = pd.DataFrame.from_csv(csv_file)
                self.reader = conf_dict['reader']

                if not isdir(self.input_dir):
                    print("Input {} is not a directory!".format(self.input_dir))
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
            print("Missing config entry {}".format(e))
            exit()

    def run(self):

        # Get paths to all images
        im_files = find_images(join(self.input_dir))
        assert (len(im_files) > 0)

        if 'augmentation' in list(self.pipeline.keys()):
            print("Starting preprocessing ({} processes)".format(self.processes))
            optimization_pool = Pool(self.processes)
            subprocess = partial(preprocess, params=self)
            results = optimization_pool.map(subprocess, im_files)
        else:
            print("Using previously augmented data")

        # Create training and validation (imbalanced)
        print("Splitting into training/validation")

        try:
            train_imgs, val_imgs = self.train_val_split(listdir(self.augment_dir))
            self.random_sample(train_imgs, val_imgs, classes=DEFAULT_CLASSES)
        except AssertionError:
            print("No images found in one more classes - unable to split training and validation")
            exit()

    def train_val_split(self, classes):

        train_imgs = [[], [], []]
        val_imgs = [[], [], []]

        for cidx, class_ in enumerate(classes):

            # Get all augmented images per class
            aug_imgs = find_images(join(self.augment_dir, class_))
            assert(len(aug_imgs) > 0)

            # Create dataframe
            patient_metadata = [image_to_metadata(img) for img in aug_imgs]
            patient_metadata = pd.DataFrame(data=patient_metadata)

            # Group images by patient and sorted by total images per patient
            grouped = [(data, len(data)) for _, data in patient_metadata.groupby('subjectID')]
            grouped = sorted(grouped, key=lambda x: x[1], reverse=True)

            # Calculate how many patients to add to training
            total_images = len(aug_imgs)
            no_train_imgs = np.floor(float(total_images) * self.pipeline.train_split)
            cum_sum = np.cumsum([g[1] for g in grouped])
            no_train_patients = next(x[0] for x in enumerate(cum_sum) if x[1] > no_train_imgs)

            # Create validation and training
            for idx, group in enumerate(grouped):

                if idx >= no_train_patients:
                    val_imgs[cidx].extend([x['image'] for x in group[0].to_dict(orient='records')])
                else:
                    train_imgs[cidx].extend([x['image'] for x in group[0].to_dict(orient='records')])

        # Ensure that we ended up with some data in both groups
        assert(all(len(tr_class) > 0 for tr_class in train_imgs))
        assert(all(len(v_class) > 0 for v_class in val_imgs))

        return train_imgs, val_imgs

    def random_sample(self, train_imgs, val_imgs, classes=DEFAULT_CLASSES):

        train_class_sizes = [len(x) for x in train_imgs]
        val_class_sizes = [len(x) for x in val_imgs]

        train_arr, val_arr = [], []
        train_labels, val_labels = [], []

        for cidx, class_ in enumerate(classes):

            train_class_dir = join(self.train_dir, class_)
            val_class_dir = join(self.val_dir, class_)

            removal_num = train_class_sizes[cidx] - int(
                (float(min(train_class_sizes)) / float(train_class_sizes[cidx])) * train_class_sizes[cidx])

            if removal_num > 0:
                random_train = self.choose_random(train_imgs[cidx], removal_num)
                train_imgs[cidx] = random_train

            for ti in train_imgs[cidx]:
                train_arr.append(np.asarray(Image.open(ti)))
                train_labels.append(class_)
                # copy(ti, train_class_dir)

            removal_num = val_class_sizes[cidx] - int(
                (float(min(val_class_sizes)) / float(val_class_sizes[cidx])) * val_class_sizes[cidx])

            if removal_num > 0:
                random_val = self.choose_random(val_imgs[cidx], removal_num)
                val_imgs[cidx] = random_val

            for vi in val_imgs[cidx]:
                val_arr.append(np.asarray(Image.open(vi)))
                val_labels.append(class_)
                # copy(vi, val_class_dir)

            print('\n---')
            print("Training ({}): {}".format(class_, len(train_imgs[cidx])))
            print("Validation ({}): {}".format(class_, len(val_imgs[cidx])))

        # Save results
        train_data = np.transpose(np.asarray(train_arr), (0, 3, 2, 1))
        val_data = np.transpose(np.asarray(val_arr), (0, 3, 2, 1))

        train_labels = np.asarray(train_labels)
        val_labels = np.asarray(val_labels)

        with h5py.File(join(self.out_dir, 'train.h5'), "w") as f:
            f.create_dataset('data', data=train_data, dtype=train_data.dtype)
            f.create_dataset('labels', data=train_labels, dtype=train_labels.dtype)

        with h5py.File(join(self.out_dir, 'val.h5'), "w") as f:
            f.create_dataset('data', data=val_data, dtype=val_data.dtype)
            f.create_dataset('labels', data=val_labels, dtype=val_labels.dtype)

    def choose_random(self, imgs, to_remove):

        return np.random.choice(imgs, len(imgs) - to_remove, replace=False)

    def preserve_originals(self, imgs, to_remove):

        # Sort the augmented images alphabetically and split into chunks (of augment_size)
        imgs = sorted(imgs)
        assert(len(imgs) % self.augment_size == 0)
        unique_chunks = [imgs[i:i+self.augment_size] for i in range(0, len(imgs), self.augment_size)]

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


def preprocess(im, params):

    print("Preprocessing {}".format(im))

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

    if params.class_merge:
        class_ = params.class_merge[class_]

    # Resize, preprocess and augment
    try:
        im_arr = cv2.imread(im)[:, :, ::-1]
    except TypeError:
        print("Error loading '{}'".format(im))
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

    # Get the labels

    class_dir = join(params.augment_dir, class_)  # this should already exist

    if params.augmenter:

        img = np.expand_dims(np.transpose(preprocessed_im, (2, 0, 1)), 0)

        i = 0
        for _ in params.augmenter.flow(img, batch_size=1, save_to_dir=class_dir,
                                       save_prefix=meta['prefix'], save_format='png'):
            i += 1
            if i >= params.augment_size:
                break

    else:

        img = preprocessed_im
        flip_names = ['noflip', 'flip']
        rotate_names = ['0', '90', '180', '270']

        for flip in [0, 1]:
            for rotate in [0, 1, 2, 3]:
                new_img = np.copy(img)
                if rotate > 0:
                    for i in range(rotate):
                        new_img = np.rot90(new_img)
                if flip == 1:
                    new_img = np.fliplr(new_img)
                im = Image.fromarray(new_img)

                out_name = '{}_{}_{}.png'.format(meta['prefix'], flip_names[flip], rotate_names[rotate])
                out_path = join(class_dir, out_name)
                im.save(out_path)

    return True

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)

    args = parser.parse_args()

    p = Pipeline(args.config)
    p.run()