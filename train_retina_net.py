#!/usr/bin/env python

from os import listdir, chdir
from os.path import dirname, basename, splitext, abspath

from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
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
        type_, weights = network['type'].lower(), network['weights']

        if 'vgg' in type_:

            from keras.applications.vgg16 import VGG16
            print "Loading pre-trained VGG model"
            self.model = VGG16(weights='imagenet', input_shape=(3, 256, 256), include_top=False)

        elif 'resnet' in type_:

            from keras.applications.resnet50 import ResNet50
            print "Loading pre-trained ResNet"
            self.model = ResNet50(weights='imagenet', input_shape=(3, 256, 256), include_top=False)

        plot(self.model, join(self.experiment_dir, 'model_{}.png'.format(type_)))

    def train(self, scratch=False):

        if scratch:
            raise NotImplementedError("Not implemented")
        else:

            # Extract features from training/testing data
            train_features, train_labels = self.predict()
            val_features, val_labels = self.predict(train=False)

            # Train our next top model...
            self.train_top_model(train_features, val_features, train_labels, val_labels)

    def predict(self, train=True):

        data_path, suffix = (self.train_dir, 'train') if train else (self.val_dir, 'validation')

        features_out = join(self.experiment_dir, 'cnn_features_{}.npy'.format(suffix))
        labels_out = join(self.experiment_dir, 'labels_{}.npy'.format(suffix))

        no_examples = len(find_images(join(data_path, '*')))

        if not (isfile(features_out) and isfile(labels_out)):

            generator = self.create_generator(data_path)

            print "Performing forward pass to generate {} features".format(suffix)
            cnn_features = self.model.predict_generator(generator, no_examples)  # are these always flattened?
            labels = generator.classes
            np.save(open(features_out, 'w'), cnn_features)
            np.save(open(labels_out, 'w'), labels)

        return features_out, labels_out

    def train_top_model(self, train_path, val_path, train_labels, val_labels):

        # Load data and labels (converting the latter to categorical)
        train_data, train_labels = np.load(open(train_path)), to_categorical(np.load(open(train_labels)))
        val_data, val_labels = np.load(open(val_path)), to_categorical(np.load(open(val_labels)))

        print "Train features shape: {}".format(train_data.shape)

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        plot(model, join(self.experiment_dir, 'top_model.png'))

        model.fit(train_data, train_labels,
                  nb_epoch=self.epochs, batch_size=32,
                  validation_data=(val_data, val_labels))

        model.save_weights(join(self.experiment_dir, 'best_weights.h5'))

    def create_generator(self, data_path):

        input_shape = self.model.input_shape[2:]

        datagen = ImageDataGenerator()
        generator = datagen.flow_from_directory(
            data_path,
            target_size=input_shape,
            batch_size=32,
            class_mode=None,
            shuffle=False)

        return generator

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    args = parser.parse_args()

    r = RetiNet(args.config)
    r.train(scratch=False)
