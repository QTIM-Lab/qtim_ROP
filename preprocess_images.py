#!/usr/bin/env python

from os import listdir, remove
from os.path import join, isdir, basename, abspath, dirname
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from functools import partial
from shutil import move

import yaml
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from common import find_images, make_sub_dir
from methods import *
from mask_retina import *

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
        self.augmenter = ImageDataGenerator(
            width_shift_range=float(self.resize['width']) * 1e-4,
            height_shift_range=float(self.resize['height']) * 1e-4,
            zoom_range=0.05, horizontal_flip=True, vertical_flip=True, fill_mode='constant')

        # Number of processes
        self.processes = int(cpu_count() * .7)

    def _parse_config(self, config):

        try:
            with open(config, 'rb') as c:

                conf_dict = yaml.load(c)
                self.input_dir = abspath(join(dirname(config), conf_dict['input_dir']))
                self.out_dir = make_sub_dir(dirname(config), splitext(basename(config))[0])

                if not isdir(self.input_dir):
                    print "Input {} is not a directory!".format(self.input_dir)
                    exit()

                # Extract pipeline parameters or set defaults
                options = conf_dict['pipeline']
                self.resize = options['resize']
                self.train_split = options['train_split']
                self.augment_size = options['augment_size']

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

        # Create training and validation (imbalanced)
        print "Splitting into training/validation"
        train_imgs, val_imgs = self.train_val_split()
        self.random_sample(train_imgs, val_imgs)

    def train_val_split(self):

        train_imgs = [[], [], []]
        val_imgs = [[], [], []]

        for cidx, class_ in enumerate(CLASSES):

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
                    val_imgs[cidx].extend([x['image'] for x in group[0].to_dict(orient='records')])
                else:
                    train_imgs[cidx].extend([x['image'] for x in group[0].to_dict(orient='records')])

        return train_imgs, val_imgs

    def random_sample(self, train_imgs, val_imgs):

        train_class_sizes = [len(x) for x in train_imgs]
        val_class_sizes = [len(x) for x in val_imgs]

        for cidx, class_ in enumerate(CLASSES):

            train_class_dir = join(self.train_dir, class_)
            val_class_dir = join(self.val_dir, class_)

            removal_num = train_class_sizes[cidx] - int(
                (float(min(train_class_sizes)) / float(train_class_sizes[cidx])) * train_class_sizes[cidx])

            if removal_num > 0:
                train_imgs[cidx] = self.preserve_originals(train_imgs[cidx], removal_num)

            for ti in train_imgs[cidx]:
                move(ti, train_class_dir)

            removal_num = val_class_sizes[cidx] - int(
                (float(min(val_class_sizes)) / float(val_class_sizes[cidx])) * val_class_sizes[cidx])

            if removal_num > 0:
                val_imgs[cidx] = self.preserve_originals(val_imgs[cidx], removal_num)

            for vi in val_imgs[cidx]:
                move(vi, val_class_dir)

            print '---'
            print "Training ({}): {}".format(class_, len(train_imgs[cidx]))
            print "Validation ({}): {}".format(class_, len(val_imgs[cidx]))

    def preserve_originals(self, imgs, to_remove):

        # Sort the augmented images alphabetically and split into chunks (of augment_size)
        imgs = sorted(imgs)
        unique_chunks = [imgs[i:i+self.augment_size] for i in xrange(0, len(imgs), self.augment_size)]

        # Calculate how many images we need to remove in each chunk
        total_proportion = float(to_remove) / float(len(imgs))
        remove_per_chunk = int(len(unique_chunks[0]) * total_proportion)

        # Loop through each chunk and sample the images needed
        subsampled = []
        for chunk in unique_chunks:

            # 4. Randomly sample the chunk for image to keep
            sub_chunk = np.random.choice(chunk, self.augment_size - remove_per_chunk, replace=False)
            subsampled.extend(sub_chunk)

        return subsampled


def preprocess(im, params):

    print "Preprocessing {}".format(im)

    # Extract metadata
    meta = image_to_metadata(im)

    # Resize, preprocess and augment
    im_arr = cv2.imread(im)[:, :, ::-1]

    # Resize and preprocess
    resized_im = imresize(im_arr, (params.resize['width'], params.resize['height']), interp='bilinear')
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
