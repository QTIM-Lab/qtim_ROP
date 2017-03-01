#!/usr/bin/env python

from os import listdir, remove
from os.path import join, isdir, basename, splitext, dirname
from multiprocessing.pool import Pool
from functools import partial
from collections import defaultdict

import yaml
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from common import find_images, make_sub_dir
from methods import *
from segmentation import SegmentUnet
from retinaunet.lib.extract_patches import visualize

CLASSES = ['No', 'Pre-Plus', 'Plus']


class Pipeline(object):

    def __init__(self, config):

        self._parse_config(config)

        # Calculate class distribution and train/val split
        self.class_distribution = {c: len(listdir(join(self.input_dir, c))) for c in CLASSES}
        largest_class = max(self.class_distribution, key=lambda k: self.class_distribution[k])
        class_size = self.class_distribution[largest_class]

        # Make directories for intermediate steps
        self.augment_dir = make_sub_dir(self.out_dir, 'augmented', tree=self.input_dir)
        self.train_dir = make_sub_dir(self.out_dir, 'training', tree=self.input_dir)
        self.val_dir = make_sub_dir(self.out_dir, 'validation', tree=self.input_dir)

        # Create augmenter
        self.augmenter = ImageDataGenerator(rotation_range=5, width_shift_range=float(self.resize['width']) * 1e-5,
            height_shift_range=float(self.resize['height']) * 1e-5, zoom_range=0.05, horizontal_flip=True,
            vertical_flip=True, fill_mode='constant')

        self.processes = 30

    def _parse_config(self, config):

        try:
            with open(config, 'rb') as c:

                conf_dict = yaml.load(c)
                self.input_dir = join(dirname(config), conf_dict['input_dir'])
                self.out_dir = make_sub_dir(dirname(config), splitext(basename(config))[0])

                if not isdir(self.input_dir):
                    print "Input {} is not a directory!".format(self.input_dir)
                    exit()

                # Extract pipeline parameters or set defaults
                options = conf_dict['pipeline']
                self.resize = options['resize']
                self.train_split = options['train_split']
                self.augment_size = options.get('augment_size', 20)

        except KeyError as e:
            print "Invalid config entry {}".format(e)
            exit()

    def run(self):

        # Get paths to all images
        im_files = find_images(join(self.input_dir, '*'))
        assert (len(im_files) > 0)

        print "Staring preprocessing ({} processes)".format(self.processes)
        optimization_pool = Pool(self.processes)
        subprocess = partial(preprocess, params=self)
        results = optimization_pool.map(subprocess, im_files)

        if not all(results):
            print "Some images failed to process..."

        # Create training and validation (imbalanced)
        train_imgs, val_imgs = self.train_val_split()
        self.random_sample(train_imgs, val_imgs)

    def train_val_split(self):

        train_imgs, val_imgs = defaultdict(list), defaultdict(list)

        for class_ in CLASSES:

            # Get all augmented images per class
            aug_imgs = find_images(join(self.augment_dir, class_))

            # Create dataframe
            patient_metadata = [image_to_metadata(img) for img in aug_imgs]
            patient_metadata = pd.DataFrame(data=patient_metadata)

            # Group images by patient and sorted by total images per patient
            grouped = [(data, len(data)) for _, data in patient_metadata.groupby('subjectID')]
            grouped = sorted(grouped, key=lambda x: x[1])

            # Calculate how many patients to add to validation before switching to training
            total_images = len(aug_imgs)
            no_val_imgs = np.ceil(float(total_images) * (1.0 - self.train_split))
            cum_sum = np.cumsum([g[1] for g in grouped])
            no_val_patients = next(x[0] for x in enumerate(cum_sum) if x[1] > no_val_imgs)

            # Create validation and training sets
            for idx, group in enumerate(grouped):

                if idx < no_val_patients:
                    val_imgs[class_].extend([x['image'] for x in group[0].to_dict(orient='records')])
                else:
                    train_imgs[class_].extend([x['image'] for x in group[0].to_dict(orient='records')])

        return train_imgs, val_imgs

    def random_sample(self, train_imgs, val_imgs):

        train_class_sizes = [len(x) for x in train_imgs.values()]
        val_class_sizes = [len(x) for x in val_imgs.values()]

        train_sample = dict

        for class_idx, class_ in enumerate(CLASSES):

            removal_num = train_class_sizes[class_idx] - int(
                (float(min(train_class_sizes)) / float(train_class_sizes[class_idx])) * train_class_sizes[class_idx])

            if removal_num > 0:
                removed_images = np.random.choice(train_imgs[class_], removal_num, replace=False)
                train_imgs[class_] = list(set(train_imgs[class_]) - set(removed_images))

            removal_num = val_class_sizes[class_idx] - int(
                (float(min(val_class_sizes)) / float(val_class_sizes[class_idx])) * val_class_sizes[class_idx])

            if removal_num > 0:
                removed_images = np.random.choice(val_imgs[class_], removal_num, replace=False)
                val_imgs[class_] = list(set(val_imgs[class_]) - set(removed_images))

            print "Training ({}): {}".format(class_, len(train_imgs[class_]))
            print "Validation ({}): {}".format(class_, len(val_imgs[class_]))

def preprocess(im, params):

    # Extract metadata
    meta = image_to_metadata(im)

    # Resize, preprocess and augment
    im_arr = cv2.imread(im)[:, :, ::-1]
    resized_im = imresize(im_arr, (params.resize['width'], params.resize['height']), interp='bicubic')
    preprocessed_im = normalize_channels(resized_im)

    img = np.expand_dims(np.transpose(preprocessed_im, (2, 0, 1)), 0)
    class_dir = join(params.augment_dir, meta['class'])  # this should already exist

    i = 0
    for _ in params.augmenter.flow(img, batch_size=1, save_to_dir=class_dir, save_prefix=meta['prefix'], save_format='bmp'):
        i += 1
        if i >= params.augment_size:
            break

    return True


def image_to_metadata(im_path):

    im_name = basename(im_path)
    im_str = splitext(im_name)[0]
    subject_id, _, im_id, session, _, eye, class_ = im_str.split('_')[:7]
    return {'subjectID': subject_id, 'session': session, 'eye': eye, 'class': class_, 'image': im_path, 'prefix': im_str}


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)

    args = parser.parse_args()

    p = Pipeline(args.config)
    p.run()
