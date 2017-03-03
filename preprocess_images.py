#!/usr/bin/env python

from os import listdir, remove
from os.path import join, isdir, basename, abspath, dirname
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from functools import partial
from shutil import copy
import itertools

import yaml
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from common import find_images, make_sub_dir
from methods import *
from mask_retina import *


class Pipeline(object):

    def __init__(self, config):

        self._parse_config(config)

        # Calculate class distribution and train/val split
        self.class_distribution = {c: len(listdir(join(self.input_dir, c))) for c in ['No', 'Pre-Plus', 'Plus']}
        self.classes = sorted(self.class_distribution, key=self.class_distribution.get)

        largest_class = max(self.class_distribution, key=lambda k: self.class_distribution[k])
        class_size = self.class_distribution[largest_class]

        # Make directories for intermediate steps
        self.augment_dir = make_sub_dir(self.out_dir, 'augmented', tree=self.input_dir)
        self.train_dir = make_sub_dir(self.out_dir, 'training', tree=self.input_dir)
        self.val_dir = make_sub_dir(self.out_dir, 'validation', tree=self.input_dir)

        # Create augmenter
        if self.augment_method == 'keras':
            self.augmenter = ImageDataGenerator(
                width_shift_range=float(self.resize['width']) * 1e-4,
                height_shift_range=float(self.resize['height']) * 1e-4,
                zoom_range=0.05, horizontal_flip=True, vertical_flip=True, fill_mode='constant')
        else:
            self.augmenter = None

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

                self.augment_method = options['augment_method']
                self.augment_size = options['augment_size']

        except KeyError as e:
            print "Invalid config entry {}".format(e)
            exit()

    def run(self):

        # Get paths to all images
        im_files = find_images(join(self.input_dir, '*'))
        assert (len(im_files) > 0)

        if self.augment_method is not None:
            print "Starting preprocessing ({} processes)".format(self.processes)
            optimization_pool = Pool(self.processes)
            subprocess = partial(preprocess, params=self)
            results = optimization_pool.map(subprocess, im_files)
        else:
            print "Using previously augmented data"

        # Create training and validation (imbalanced)
        print "Splitting into training/validation"
        train_imgs, val_imgs = self.train_val_split()
        self.random_sample(train_imgs, val_imgs)

    def train_val_split(self):

        train_imgs = [[], [], []]
        val_imgs = [[], [], []]

        for cidx, class_ in enumerate(self.classes):

            # Get all augmented images per class
            aug_imgs = find_images(join(self.augment_dir, class_))

            # Create dataframe
            patient_metadata = [image_to_metadata(img) for img in aug_imgs]
            patient_metadata = pd.DataFrame(data=patient_metadata)

            # Group images by patient and sorted by total images per patient
            grouped = [(data, len(data)) for _, data in patient_metadata.groupby('subjectID')]
            grouped = sorted(grouped, key=lambda x: x[1])

            print "Class: {}".format(class_)
            print "Number of subjects: {}".format(len(grouped))

            # Calculate how many patients to add to each group
            total_images = len(aug_imgs)
            no_val_imgs = np.round(float(total_images) * (1.0 - self.train_split))
            no_train_imgs = np.round(float(total_images) * (self.train_split))

            # Generate all possible permutations of the grouped data
            best_error = np.iinfo(np.int16)
            best_t, best_v = None, None

            for perm in itertools.permutations(range(0, len(grouped))):

                # Permute the groups
                perm_g = [grouped[i] for i in perm]

                # Try to assign all possible subsets of this permutation to training/validation
                for i in range(1, len(perm) - 1):

                    t_sub = perm_g[:i]  # training subset
                    v_sub = perm_g[i:]  # validation subset

                    t_err = abs(sum([g[1] for g in t_sub]) - no_train_imgs)
                    v_err = abs(sum([g[1] for g in v_sub]) - no_val_imgs)
                    total_err = t_err + v_err  # how far off we from having a balance training/validation

                    if total_err < best_error:  # keep this t/v division if better than previous
                        best_error = total_err
                        best_t = t_sub
                        best_v = v_sub

            # Create validation and training sets based on optimal split
            for idx, group in enumerate(best_t):
                train_imgs[cidx].extend([x['image'] for x in group[0].to_dict(orient='records')])

            for idx, group in enumerate(best_v):
                val_imgs[cidx].extend([x['image'] for x in group[0].to_dict(orient='records')])

        # Ensure that we ended up with some data in both groups
        assert(all(len(tr_class) > 0 for tr_class in train_imgs))
        assert(all(len(v_class) > 0 for v_class in val_imgs))

        return train_imgs, val_imgs

    def random_sample(self, train_imgs, val_imgs):

        train_class_sizes = [len(x) for x in train_imgs]
        val_class_sizes = [len(x) for x in val_imgs]

        for cidx, class_ in enumerate(self.classes):

            train_class_dir = join(self.train_dir, class_)
            val_class_dir = join(self.val_dir, class_)

            removal_num = train_class_sizes[cidx] - int(
                (float(min(train_class_sizes)) / float(train_class_sizes[cidx])) * train_class_sizes[cidx])

            if removal_num > 0:
                train_imgs[cidx] = self.preserve_originals(train_imgs[cidx], removal_num)

            for ti in train_imgs[cidx]:
                copy(ti, train_class_dir)

            removal_num = val_class_sizes[cidx] - int(
                (float(min(val_class_sizes)) / float(val_class_sizes[cidx])) * val_class_sizes[cidx])

            if removal_num > 0:
                val_imgs[cidx] = self.preserve_originals(val_imgs[cidx], removal_num)

            for vi in val_imgs[cidx]:
                copy(vi, val_class_dir)

            print '\n---'
            print "Training ({}): {}".format(class_, len(train_imgs[cidx]))
            print "Validation ({}): {}".format(class_, len(val_imgs[cidx]))

    def preserve_originals(self, imgs, to_remove):

        # Sort the augmented images alphabetically and split into chunks (of augment_size)
        imgs = sorted(imgs)
        assert(len(imgs) % self.augment_size == 0)
        unique_chunks = [imgs[i:i+self.augment_size] for i in xrange(0, len(imgs), self.augment_size)]

        # Calculate how many images we need to remove in each chunk (some chunks likely smaller than others)
        total_proportion = float(to_remove) / float(len(imgs))
        remove_per_chunk = [int(np.ceil(len(x) * total_proportion)) for x in unique_chunks]

        # Loop through each chunk and sample the images needed
        subsampled = []
        for chunk, to_remove in zip(unique_chunks, remove_per_chunk):

            # Randomly sample the chunk for image to keep
            sub_chunk = np.random.choice(chunk, len(chunk) - to_remove, replace=False)
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

    class_dir = join(params.augment_dir, meta['class'])  # this should already exist

    if params.augmenter:

        img = np.expand_dims(np.transpose(preprocessed_im, (2, 0, 1)), 0)

        i = 0
        for _ in params.augmenter.flow(img, batch_size=1, save_to_dir=class_dir, save_prefix=meta['prefix'], save_format='jpg'):
            i += 1
            if i >= params.augment_size:
                break

    else:

        img = preprocessed_im
        outfile = join(class_dir, meta['prefix'] + '.jpg')
        flipnames = ['noflip', 'flip']
        rotatenames = ['0', '90', '180', '270']

        for flip in [0, 1]:
            for rotate in [0, 1, 2, 3]:
                new_img = np.copy(img)
                if rotate > 0:
                    for i in xrange(rotate):
                        new_img = np.rot90(new_img)
                if flip == 1:
                    new_img = np.fliplr(new_img)
                im = Image.fromarray(new_img)
                splitfile = str.split(outfile, '.')
                new_outfile = splitfile[0] + '_' + flipnames[flip] + '_' + rotatenames[rotate] + '.' + splitfile[-1]
                im.save(new_outfile)

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
