from keras.models import Model
from keras.layers import Input, Convolution2D, Dropout, ZeroPadding2D, MaxPooling2D, Flatten, Dense
from keras.utils.visualize_util import plot


def simple_CNN():

    input = Input(shape=(3, 256, 256))

    conv_1 = Convolution2D(32, 3, 3, activation='relu')(input)
    pad_1 = ZeroPadding2D()(conv_1)

    conv_2 = Convolution2D(64, 3, 3, activation='relu')(pad_1)
    pad_2 = ZeroPadding2D()(conv_2)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(pad_2)
    drop_1 = Dropout(.5)(pool_1)

    conv_3 = Convolution2D(64, 3, 3, activation='relu')(drop_1)
    pad_3 = ZeroPadding2D()(conv_2)
    conv_4 = Convolution2D(64, 3, 3, activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    drop_2 = Dropout(.5)(pool_2)

    flat = Flatten()(drop_2)
    hidden = Dense(512, activation='relu')(flat)
    drop_3 = Dropout(.25)(hidden)
    output = Dense(3, activation='softmax')(drop_3)

    model = Model(input=input, output=output) # To define a model, just specify its input and output layers

    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['accuracy']) # reporting the accuracy

    return model

if __name__ == '__main__':

    model = simple_CNN()
    plot(model, to_file='arch.png', show_shapes=True)