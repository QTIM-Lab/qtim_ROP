from PIL import Image
from glob import glob
import numpy as np
from scipy.misc import imresize
from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD


def main(X, y, channels, normalize=False):

    # Split training data
    X_train, X_test = train_test_split(X, test_size=.2)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Normalise to 0. - 1.
    if normalize:
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.

    # Train model
    print("Training autoencoder")
    model = train_autoencoder(X_train, X_test, channels)


def train_autoencoder(X_train, X_test, channels):

    # Create an autoencoder
    model = autoencoder(channels=channels)
    model.fit(X_train, X_train, nb_epoch=100, batch_size=12, shuffle=True, validation_data=(X_test, X_test))
    return model


def load_data(img_dir, csv_file):

    # Load images
    img_list = sorted(glob(join(img_dir, '*')))
    X = np.asarray([imresize(np.asarray(Image.open(img)), (256, 256)) for img in img_list])

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=3)
        channels = 1
    else:
        channels = 3

    X = np.transpose(X, (0, 3, 1, 2))

    # Load labels
    raw_classes = pd.DataFrame.from_csv(csv_file)['class_name']

    class_indices = {k: v for v, k in enumerate(np.unique(raw_classes))}
    classes = [class_indices[k] for k in raw_classes]
    y = to_categorical(classes)

    return X, y, channels


def autoencoder(channels=3):

    input_img = Input(shape=(channels, 256, 256))

    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(channels, 3, 3, activation='sigmoid', border_mode='same')(x)

    ae = Model(input_img, decoded)

    # sgd = SGD(lr=0.001, momentum=.9, decay=1e-3)
    ae.compile(optimizer='adadelta', loss='mse')
    return ae

if __name__ == '__main__':

    import sys

    imgs, labels, channels = load_data(sys.argv[1], sys.argv[2])
    main(imgs, labels, channels)