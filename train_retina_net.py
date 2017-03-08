#!/usr/bin/env python

import sys
from os import listdir, chdir
from os.path import dirname, basename, splitext, abspath
import logging
from shutil import copy
from sklearn.metrics import confusion_matrix

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

from common import *
from plotting import *


class RetiNet(object):

    def __init__(self, conf_file):

        conf_dir = dirname(conf_file)
        experiment_name = splitext(basename(conf_file))[0]

        if isdir(join(conf_dir, experiment_name)):
            print "Folder '{}' already exists!".format(experiment_name)
            print "Please rename the YAML file, or delete the existing data."
            exit()

        # Parse config and create output dir
        self.config = parse_yaml(conf_file)
        self.experiment_dir = make_sub_dir(conf_dir, experiment_name)
        copy(conf_file, self.experiment_dir)

        # Set up logging
        setup_log(self.experiment_dir, to_file=self.config.get('logging', False))
        logging.info("Experiment name: {}".format(experiment_name))

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
        fine_tuning = " with pre-trained weights '{}'".format(weights) if weights else " without pre-training"

        if 'vgg' in type_:

            from keras.applications.vgg16 import VGG16
            logging.info("Instantiating VGG model" + fine_tuning)
            self.model = VGG16(weights=weights, input_shape=(3, 227, 227), include_top=True)

        elif 'resnet' in type_:

            from keras.applications.resnet50 import ResNet50
            logging.info("Instantiating ResNet model" + fine_tuning)
            self.model = ResNet50(weights=weights, input_shape=(3, 256, 256), include_top=True)

        elif 'googlenet' in type_:

            from googlenet_custom_layers import PoolHelper, LRN
            from keras.models import model_from_json

            logging.info("Instantiating GoogLeNet model" + fine_tuning)
            arch = network.get('arch', None)
            self.model = model_from_json(open(arch).read(), custom_objects={"PoolHelper": PoolHelper, "LRN": LRN})
            if weights:
                self.model.load_weights(weights, by_name=True)  # TODO check this second argument
            self.model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        else:
            raise KeyError("Invalid network type '{}'".format(type_))

        plot(self.model, join(self.experiment_dir, 'model_architecture{}.svg'.format(type_)))

    def train(self):

        # Train
        input_shape = self.model.input_shape[1:]
        train_gen = self.create_generator(self.train_dir, input_shape, mode='categorical')
        val_gen = self.create_generator(self.val_dir, input_shape, mode='categorical')

        # Make callbacks
        weights_out = join(self.experiment_dir, 'best_weights.h5')
        checkpoint_tb = ModelCheckpoint(filepath=weights_out, verbose=1, save_best_only=True)

        logging.info("Training model for {} epochs".format(self.epochs))
        history = self.model.fit_generator(
            train_gen,
            samples_per_epoch=self.nb_train_samples,
            nb_epoch=self.epochs,
            validation_data=val_gen,
            nb_val_samples=self.nb_val_samples, callbacks=[checkpoint_tb])

        # Save final weights
        self.model.save_weights(join(self.experiment_dir, 'final_weights.h5'))

        # Predict validation data
        predictions = self.model.predict_generator(val_gen, self.nb_val_samples)
        y_true, y_pred = val_gen.classes, np.argmax(predictions, axis=1)
        labels = [k[0] for k in sorted(val_gen.class_indices.items(), key=lambda x: x[1])]
        confusion = confusion_matrix(y_true, y_pred)

        plot_accuracy(history, join(self.experiment_dir, 'accuracy.svg'))
        plot_loss(history, join(self.experiment_dir, 'loss.svg'))
        plot_confusion(confusion, labels, join(self.experiment_dir, 'confusion.svg'))

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
