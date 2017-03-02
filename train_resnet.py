from os import listdir
from os.path import dirname, basename, splitext, abspath

from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from resnet50 import ResNet50

from common import *


class RetinaResNet(object):

    def __init__(self, conf_file):

        # Parse config and create output dir
        config = parse_yaml(conf_file)
        conf_dir = dirname(conf_file)
        experiment_name = splitext(basename(conf_file))[0]
        self.experiment_dir = make_sub_dir(conf_dir, experiment_name)
        self.train_dir = abspath(join(conf_dir, config['training_dir']))
        self.val_dir = abspath(join(conf_dir, config['validation_dir']))

        self.no_classes = listdir(self.train_dir)
        self.nb_train_samples = len(find_images(join(self.train_dir, '*')))
        self.nb_val_samples = len(find_images(join(self.val_dir, '*')))

    def train(self):

        # Create ResNet
        print "Loading pre-trained ResNet"
        model = ResNet50(weights='imagenet')
        plot(model, join(self.experiment_dir, 'model_full.png'))

        # Pop the last (Dense) and penultimate (Flatten) layers off
        model.layers.pop()
        plot(model, join(self.experiment_dir, 'model_popped.png'))

        # Extract features from training/testing data
        train_features, train_labels = self.predict(model)
        val_features, val_labels = self.predict(model, train=False)

        # Train our next top model...
        self.train_top_model(train_features, val_features, train_labels, val_labels)

    def predict(self, model, train=True):

        data_path, suffix = (self.train_dir, 'train') if train else (self.val_dir, 'val')
        features_out = join(self.experiment_dir, 'cnn_features_{}.npy'.format(suffix))
        labels_out = join(self.experiment_dir, 'labels_{}.npy'.format(suffix))
        no_examples = len(find_images(join(data_path, '*')))

        if not (isfile(features_out) and isfile(labels_out)):

            datagen = ImageDataGenerator()
            generator = datagen.flow_from_directory(
                data_path,
                target_size=model.input_shape[2:],
                batch_size=32,
                class_mode=None,
                shuffle=False)

            print "Performing forward pass to generate CNN features"
            cnn_features = model.predict_generator(generator, no_examples)  # are these always flattened?
            labels = generator.classes
            np.save(open(features_out, 'w'), cnn_features)
            np.save(open(labels_out, 'w'), labels)

        return features_out, labels_out

    def train_top_model(self, train_path, val_path, train_labels, val_labels, nb_epoch=50):

        # Load data and labels (converting the latter to categorical)
        train_data, train_labels = np.load(open(train_path)), to_categorical(np.load(open(train_labels)))
        val_data, val_labels = np.load(open(val_path)), to_categorical(np.load(open(val_labels)))

        print "Train features shape: {}".format(train_data.shape)

        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=train_data.shape[1:]))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        plot(model, join(self.experiment_dir, 'top_model.png'))

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
