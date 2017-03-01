from os import mkdir, listdir
from os.path import dirname, basename, splitext

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from resnet50 import ResNet50

from common import *

class RetinaResNet(object):

    def __init__(self, conf_file):

        # Parse config and create output dir
        config = parse_yaml(conf_file)
        experiment_name = splitext(basename(conf_file))[0]
        self.experiment_dir = make_sub_dir(dirname(conf_file), experiment_name)
        self.train_dir = config['training_dir']
        self.val_dir = config['validation_dir']

        self.no_classes = listdir(self.train_dir)
        self.nb_train_samples = len(find_images(join(self.train_dir, '*')))
        self.nb_val_samples = len(find_images(join(self.val_dir, '*')))


    def train(self):

        # Create ResNet
        model = ResNet50(weights='imagenet')
        plot(model, join(self.experiment_dir, 'model_full.png'))

        # Pop the last layer off
        model.layers.pop()
        plot(model, join(self.experiment_dir, 'model_popped.png'))

        # Extract features from training/testing data
        train_features, nb_train_examples = self.predict(model)
        val_features, nb_val_examples = self.predict(model, train=False)

        # Train our next top model...
        self.train_top_model(train_features, val_features, nb_train_examples, nb_val_examples)

    def predict(self, model, train=True):

        data_path, suffix = (self.train_dir, 'train') if train else (self.val_dir, 'val')

        datagen = ImageDataGenerator()
        generator = datagen.flow_from_directory(
            data_path,
            target_size=model.input_shape[2:],
            batch_size=32,
            class_mode=None,
            shuffle=False)

        no_examples = len(find_images(join(data_path, '*')))
        cnn_features = model.predict_generator(generator, no_examples)
        npy_out = join(self.experiment_dir, 'cnn_features_{}.npy'.format(suffix))
        np.save(open(npy_out, 'w'), cnn_features)
        return npy_out, no_examples

    def train_top_model(self, train_path, val_path, nb_train_samples, nb_val_examples, nb_epoch=50):

        train_data = np.load(open(train_path))
        train_labels = np.array([0] * (nb_train_samples / self.no_classes) +
                                [1] * (nb_train_samples / self.no_classes) +
                                [2] * (nb_train_samples / self.no_classes))

        val_data = np.load(open(val_path))
        val_labels = np.array([0] * (nb_val_examples / self.no_classes) +
                              [1] * (nb_val_examples / self.no_classes) +
                              [2] * (nb_val_examples / self.no_classes))

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_data, train_labels,
                  nb_epoch=nb_epoch, batch_size=32,
                  validation_data=(val_data, val_labels))

        model.save_weights(join(self.experiment_dir, 'best_weights.h5'))

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    args = parser.parse_args()

    r = RetinaResNet(args.config)
    r.train()
