#!/usr/bin/env python

from os import listdir, chdir
from os.path import dirname, basename, splitext, abspath

from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Flatten, Activation
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

    def _configure_network(self, fine_tune=False):

        self.epochs = self.config.get('epochs', 50)
        network = self.config['network']
        type_, weights = network['type'].lower(), network.get('weights', None)

        if 'vgg' in type_:

            from keras.applications.vgg16 import VGG16
            print "Loading pre-trained VGG model"
            self.model = VGG16(weights=weights, input_shape=(3, 227, 227), include_top=not fine_tune)

        elif 'resnet' in type_:

            from keras.applications.resnet50 import ResNet50
            print "Loading pre-trained ResNet"
            self.model = ResNet50(weights=weights, input_shape=(3, 256, 256), include_top=not fine_tune)

        elif 'inception' in type_:

            from keras.applications.inception_v3 import InceptionV3
            self.model = InceptionV3(weights=weights, input_shape=(3, 299, 299), include_top=not fine_tune)

        # elif 'alex' in type_:
        #
        #     from convnetskeras.convnets import convnet
        #     alexnet = convnet('alexnet')  #, weights_path='alexnet_weights.h5')
        #
        #     input = alexnet.input
        #     img_representation = alexnet.get_layer("dense_2").output
        #
        #     classifier = Dense(3, name='classifier')(img_representation)
        #     classifier = Activation("softmax", name="softmax")(classifier)
        #     model = Model(input=input, output=classifier)
        #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
        #
        #     print model.get_layer('softmax').output_shape
        #     self.model = model

        else:
            raise KeyError("Invalid network type '{}'".format(type_))

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        plot(self.model, join(self.experiment_dir, 'model_{}.png'.format(type_)))

    def train(self, fine_tune=False):

        # Train
        input_shape = self.model.input_shape[1:]
        train_gen = self.create_generator(self.train_dir, input_shape, mode='categorical')
        val_gen = self.create_generator(self.val_dir, input_shape, mode='categorical')

        self.model.fit_generator(
            train_gen,
            samples_per_epoch=self.nb_train_samples,
            nb_epoch=self.epochs,
            validation_data=val_gen,
            nb_val_samples=self.nb_val_samples)

        self.model.save_weights(join(self.experiment_dir, 'best_weights.h5'))

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
