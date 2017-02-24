from common import find_images
from os import mkdir
from os.path import join, isdir

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def train_classifier(image_path, out_dir):

    # Grab a test image
    test_img = find_images(join(image_path, '*'))[0]
    img = load_img(test_img)  # PIL
    x = img_to_array(img)  # numpy array
    x = x.reshape((1,) + x.shape)  # numpy array with shape (1, channels, x, y)

    # Create "simple" keras CNN
    im_shape = x.shape[1:]
    model = define_model(im_shape)

    # Create data generator
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=1e-4,
        height_shift_range=1e-4,
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    # Generator for training
    target_size = im_shape[1:]
    train_generator = datagen.flow_from_directory(
        image_path,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical')

    # Fit!
    model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50)

    # Save weights
    if not isdir(out_dir):
        mkdir(out_dir)
    model.save_weights(join(out_dir, 'first_try.h5'))


# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def define_model(img_shape):

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=img_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-t', '--training', dest='training', required=True)
    parser.add_argument('-o', '--output', dest='out_dir', required=True)
    args = parser.parse_args()

    train_classifier(args.training, args.out_dir)
