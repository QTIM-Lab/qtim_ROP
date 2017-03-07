#!/usr/bin/env python

from os import listdir, chdir
from os.path import dirname, basename, splitext, abspath

from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical

from common import *


class RetiNet(object):

    def __init__(self, conf_file):

        # Parse config and create output dir
        self.config = parse_yaml(conf_file)
        conf_dir = dirname(conf_file)
        experiment_name = splitext(basename(conf_file))[0]

        # Define input and output directories
        self.experiment_dir = make_sub_dir(conf_dir, experiment_name)

        chdir(conf_dir)
        self.train_dir = abspath(self.config['training_dir'])
        self.val_dir = abspath(self.config['validation_dir'])

        # Get number of classes and samples
        self.no_classes = listdir(self.train_dir)
        self.nb_train_samples = len(find_images(join(self.train_dir, '*')))
        self.nb_val_samples = len(find_images(join(self.val_dir, '*')))

        # Create the network based on params
        self._configure_network()

    def _configure_network(self):

        self.epochs = self.config.get('epochs', 50)
        network = self.config['network']
        type_, weights = network['type'].lower(), network.get('weights', None)

        if 'vgg' in type_:

            from keras.applications.vgg16 import VGG16
            print "Loading pre-trained VGG model"
            self.model = VGG16(weights=weights, input_shape=(3, 227, 227), include_top=True)
            self.no_outputs = 1

        elif 'resnet' in type_:

            from keras.applications.resnet50 import ResNet50
            print "Loading pre-trained ResNet"
            self.model = ResNet50(weights=weights, input_shape=(3, 256, 256), include_top=True)
            self.no_outputs = 1

        elif 'googlenet' in type_:

            from googlenet_custom_layers import PoolHelper, LRN
            from keras.models import model_from_json

            arch = network.get('arch', None)
            self.model = model_from_json(open(arch).read(), custom_objects={"PoolHelper": PoolHelper, "LRN": LRN})
            self.model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
            self.no_outputs = 3

        else:
            raise KeyError("Invalid network type '{}'".format(type_))

        plot(self.model, join(self.experiment_dir, 'model_final_{}.png'.format(type_)))

    def train(self):

        # Train
        input_shape = self.model.input_shape[1:]
        train_gen = self.create_generator(self.train_dir, input_shape, mode='categorical')
        val_gen = self.create_generator(self.val_dir, input_shape, mode='categorical')

        if self.no_outputs > 1:
            val_gen = zip(*[val_gen] * self.no_outputs)  # fix to deal with three outputs

        print "Fitting..."
        self.model.fit_generator(
            train_gen,
            samples_per_epoch=self.nb_train_samples,
            nb_epoch=self.epochs,
            validation_data=val_gen,
            nb_val_samples=self.nb_val_samples)

        self.model.save_weights(join(self.experiment_dir, 'best_weights.h5'))

        # Predict validation data
        predictions = self.model.predict_generator(val_gen, self.nb_val_samples)
        print predictions

    def create_generator(self, data_path, input_shape, mode=None):

        datagen = ImageDataGenerator()
        generator = datagen.flow_from_directory(
            data_path,
            target_size=input_shape[1:],
            batch_size=32,
            class_mode=mode)

        return generator

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    args = parser.parse_args()

    r = RetiNet(args.config)
    r.train()
