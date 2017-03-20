from os.path import split, join
from keras import __version__
from keras.models import model_from_json, Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.visualize_util import plot

import json


def load_model(model):

    # Load model
    _, model_basename = split(model)
    model_arch = join(model, model_basename + '_architecture.json')
    model_weights = join(model, model_basename + '_best_weights.h5')

    model = model_from_json(open(model_arch).read())
    model.load_weights(model_weights)

    return model


def top_model(base_shape):

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_shape))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='sigmoid'))


def simple_CNN(out_dir=None):

    print __version__

    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(3, 224, 224)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    if out_dir:

        model_out = join(out_dir, 'simple_CNN.json')
        with open(model_out, 'w') as json_file:
            json_file.write(model.to_json())

        plot(model, to_file=join(out_dir, 'arch.png'), show_shapes=True)

        loaded = model_from_json(open(model_out).read())

if __name__ == '__main__':

    import sys

    simple_CNN(sys.argv[1])

