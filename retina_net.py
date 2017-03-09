#!/usr/bin/env python

import sys
from os import listdir, chdir
from os.path import dirname, basename, splitext, abspath
import logging
from shutil import copy
from sklearn.metrics import confusion_matrix

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from common import *
from plotting import *


class RetiNet(object):

    def __init__(self, conf_file):

        # Parse config
        self.conf_file = conf_file
        self.config = parse_yaml(conf_file)
        self.ext = self.config.get('ext', '.png')

        self.conf_dir = dirname(self.conf_file)
        self.experiment_name = splitext(basename(self.conf_file))[0]

        chdir(self.conf_dir)
        self.train_dir = abspath(self.config['training_dir'])
        self.val_dir = abspath(self.config['validation_dir'])

        try:
            self.config['mode']
        except KeyError:
            print "Please specify a mode 'train' or 'evaluate' in the config file."
            exit()

        if self.config['mode'] == 'train':

            # Set up logging
            self.experiment_dir = make_sub_dir(self.conf_dir, self.experiment_name)
            setup_log(join(self.experiment_dir, 'training.log'), to_file=self.config.get('logging', False))
            logging.info("Experiment name: {}".format(self.experiment_name))
            self._configure_network()

            # Get number of classes and samples
            self.no_classes = listdir(self.train_dir)
            self.nb_train_samples = len(find_images(join(self.train_dir, '*')))
            self.nb_val_samples = len(find_images(join(self.val_dir, '*')))
            self.train()

        elif self.config['mode'] == 'evaluate':

            # Set up logging
            setup_log(None)
            self._configure_network()
            self.experiment_dir = self.conf_dir
            self.evaluate(self.config['test_dir'])

    def _configure_network(self):

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

    def train(self):

        # Train
        epochs = self.config.get('epochs', 50)  # default to 50 if not specified
        input_shape = self.model.input_shape[1:]
        train_gen = self.create_generator(self.train_dir, input_shape, training=True)
        val_gen = self.create_generator(self.val_dir, input_shape, training=False)

        # Check point callback saves weights on improvement
        weights_out = join(self.experiment_dir, 'best_weights.h5')
        checkpoint_tb = ModelCheckpoint(filepath=weights_out, verbose=1, save_best_only=True)

        logging.info("Training model for {} epochs".format(epochs))
        history = self.model.fit_generator(
            train_gen,
            samples_per_epoch=self.nb_train_samples,
            nb_epoch=epochs,
            validation_data=val_gen,
            nb_val_samples=self.nb_val_samples, callbacks=[checkpoint_tb])

        # Save model arch, weighs and history
        dict_to_csv(history.history, join(self.experiment_dir, "history.csv"))
        self.model.save_weights(join(self.experiment_dir, 'final_weights.h5'))

        with open(join(self.experiment_dir, 'model_arch.json'), 'w') as arch:
            arch.writelines(self.model.to_json())

        # Create modified copy of config file
        conf_eval = self.update_config()
        with open(join(self.experiment_dir, self.experiment_name + '.yaml'), 'wb') as ce:
            yaml.dump(conf_eval, ce, default_flow_style=False)

        # Evaluate results
        self.evaluate(self.val_dir)

    def update_config(self):

        conf_eval = dict(self.config)
        conf_eval['mode'] = 'evaluate'
        conf_eval['network']['arch'] = 'model_arch.json'
        conf_eval['network']['weights'] = 'best_weights.h5'

        conf_eval['training_dir'] = abspath(self.config['training_dir'])
        conf_eval['validation_dir'] = abspath(self.config['validation_dir'])
        return conf_eval

    def evaluate(self, data_path):

        logging.info("Evaluating model for on {}".format(data_path))
        history = csv_to_dict(join(self.experiment_dir, "history.csv"))
        datagen = self.create_generator(data_path, self.model.input_shape[1:], training=False)
        no_samples = len(find_images(join(data_path, '*')))

        # Predict data
        predictions = self.model.predict_generator(datagen, no_samples)
        y_true, y_pred = datagen.classes, np.argmax(predictions, axis=1)
        labels = [k[0] for k in sorted(datagen.class_indices.items(), key=lambda x: x[1])]
        confusion = confusion_matrix(y_true, y_pred)

        # Plots
        plot_accuracy(history, join(self.experiment_dir, 'accuracy' + self.ext))
        plot_loss(history, join(self.experiment_dir, 'loss' + self.ext))
        plot_confusion(confusion, labels, join(self.experiment_dir, 'confusion' + self.ext))

    def create_generator(self, data_path, input_shape, training=True):

        zmuv = self.config.get('zmuv', False)
        if zmuv:
            logging.info('Normalizing data zero mean, unit variance')

        datagen = ImageDataGenerator(samplewise_center=zmuv, samplewise_std_normalization=zmuv)
        generator = datagen.flow_from_directory(
            data_path,
            target_size=input_shape[1:],
            batch_size=32,
            class_mode='categorical',
            shuffle=training)

        return generator

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    args = parser.parse_args()

    r = RetiNet(args.config)
