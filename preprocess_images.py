#!/usr/bin/env python

from os import mkdir, listdir
from os.path import join, isdir, basename, splitext, dirname
from random import shuffle, uniform
import yaml
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator

from common import find_images, make_sub_dir
from methods import *
from segmentation import SegmentUnet
from retinaunet.lib.extract_patches import visualize

CLASSES = ['No', 'Pre-Plus', 'Plus']
SCALE = 256


class Pipeline(object):

    def __init__(self, config):

        self._parse_config(config)

        # Calculate class distribution and train/val split
        self.class_distribution = {c: len(listdir(join(self.input_dir, c))) for c in CLASSES}
        largest_class = max(self.class_distribution, key=lambda k: self.class_distribution[k])
        class_size = self.class_distribution[largest_class]

        self.train_size = np.round(float(class_size) * self.train_split)
        self.val_size = np.round(float(class_size) * (1 - self.train_split))
        self.augment_rate = {k: int(np.ceil(1.0 / (float(v) / class_size))) for k, v in self.class_distribution.items()}

        # Make directories for intermediate steps
        self.training_dir = make_sub_dir(self.out_dir, 'training')
        self.validation_dir = make_sub_dir(self.out_dir, 'validation')

        # Create augmenter
        self.augmenter = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=float(self.resize['width']) * 1e-5,
            height_shift_range=float(self.resize['height']) * 1e-5,
            # rescale=1. / 255,
            shear_range=0.01,
            zoom_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
            cval=128)

    def _parse_config(self, config):

        try:
            with open(config, 'rb') as c:

                conf_dict = yaml.load(c)
                self.input_dir = join(dirname(config), conf_dict['input_dir'])

                if not isdir(self.input_dir):
                    print "Input {} is not a directory!".format(self.out_dir)
                    exit()

                self.out_dir = make_sub_dir(dirname(config), splitext(basename(config))[0])

                options = conf_dict['pipeline']
                self.resize, self.unet, self.train_split = options['resize'], options['unet'], options['train_split']

        except KeyError as e:
            print "Invalid config entry {}".format(e)
            exit()

    def run(self):

        for class_ in CLASSES:

            train_ids, val_ids = [], []

            im_list = find_images(join(self.input_dir, class_))
            shuffle(im_list)
            assert(len(im_list) > 0)

            for im in im_list:

                im_name = basename(im)
                metadata = image_to_metadata(im_name)
                sid = metadata['subjectID']

                # Resize, preprocess and augment
                im_arr = cv2.imread(im)
                resized_im = imresize(im_arr, (self.resize['width'], self.resize['height']), interp='bicubic')
                preprocessed_im = kaggle_BG(resized_im, 128)

                # Add to training or validation
                r = uniform(0, 1)
                if r < self.train_split and sid not in val_ids:
                    out_dir = self.training_dir
                    train_ids.append(sid)
                else:
                    out_dir = self.validation_dir
                    val_ids.append(sid)

                # Resize again and augment
                img = np.transpose(np.expand_dims(preprocessed_im, 0), (0, 3, 1, 2))
                self.augment(img, splitext(im_name)[0], out_dir, class_)

    def augment(self, img, im_name, out_dir, class_):

        class_dir = make_sub_dir(out_dir, class_)

        i = 0
        for _ in self.augmenter.flow(img, batch_size=1, save_to_dir=class_dir, save_prefix=im_name, save_format='bmp'):
            i += 1
            if i >= self.augment_rate[class_]:
                break


def image_to_metadata(im_str):

    subject_id, _, im_id, session, _, eye, class_ = im_str.split('_')[:7]
    return {'subjectID': subject_id, 'session': session, 'eye': eye, 'class': class_}

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)

    args = parser.parse_args()

    p = Pipeline(args.config)
    p.run()
