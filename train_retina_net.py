#!/usr/bin/env python

import sys
from os import listdir, chdir, devnull
from os.path import dirname, basename, splitext, abspath
import logging
from shutil import copy
from sklearn.metrics import confusion_matrix

sys.stderr = open(devnull, "w")
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
sys.stderr = sys.__stderr__

from common import *
from plotting import *


class RetiNet(object):

    def __init__(self, conf_file):

        # Parse config and create output dir
        self.config = parse_yaml(conf_file)
        conf_dir = dirname(conf_file)
        experiment_name = splitext(basename(conf_file))[0]

        # Define input and output directories
        self.experiment_dir = make_sub_dir(conf_dir, experiment_name)
        copy(conf_file, self.experiment_dir)

        if self.config.get('logging', True):
            log_path = join(self.experiment_dir, 'output.log')

            self.log_file = open(log_path, 'a')
            sys.stdout = self.log_file
            sys.stderr = self.log_file
            logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

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

        if 'vgg' in type_:

            from keras.applications.vgg16 import VGG16
            logging.info("Instantiating VGG model")
            self.model = VGG16(weights=weights, input_shape=(3, 227, 227), include_top=True)

        elif 'resnet' in type_:

            from keras.applications.resnet50 import ResNet50
            logging.info("Instantiating ResNet model")
            self.model = ResNet50(weights=weights, input_shape=(3, 256, 256), include_top=True)

        elif 'googlenet' in type_:

            from googlenet_custom_layers import PoolHelper, LRN
            from keras.models import model_from_json

            logging.info("Instantiating GoogLeNet model")
            arch = network.get('arch', None)
            self.model = model_from_json(open(arch).read(), custom_objects={"PoolHelper": PoolHelper, "LRN": LRN})
            self.model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

        else:
            raise KeyError("Invalid network type '{}'".format(type_))

        plot(self.model, join(self.experiment_dir, 'model_final_{}.png'.format(type_)))

    def train(self):

        # Train
        input_shape = self.model.input_shape[1:]
        train_gen = self.create_generator(self.train_dir, input_shape, mode='categorical')
        val_gen = self.create_generator(self.val_dir, input_shape, mode='categorical')

        # Output
        weights_out = join(self.experiment_dir, 'best_weights.h5')
        check_pointer = ModelCheckpoint(filepath=weights_out, verbose=1, save_best_only=True)

        logging.info("Fitting model to training data")
        history = self.model.fit_generator(
            train_gen,
            samples_per_epoch=self.nb_train_samples,
            nb_epoch=self.epochs,
            validation_data=val_gen,
            nb_val_samples=self.nb_val_samples, callbacks=[check_pointer])

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
