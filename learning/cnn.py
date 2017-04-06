from os.path import join

from keras.models import Model
from keras.layers import Input, Convolution2D, Dropout, ZeroPadding2D, MaxPooling2D, Flatten, Dense
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator

import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

input = Input(shape=(3, 224, 224))

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

model = Model(input=input, output=output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


plot(model, to_file=join(out_dir, 'arch.png'), show_shapes=True)


train_gen = ImageDataGenerator().flow_from_directory(join(in_dir, 'training'), target_size=(224, 224))
val_gen = ImageDataGenerator().flow_from_directory(join(in_dir, 'validation'), target_size=(224, 224))


model.fit_generator(train_gen, samples_per_epoch=train_gen.nb_sample, nb_epoch=100,
                    validation_data=val_gen, nb_val_samples=val_gen.nb_sample)

model.save(join(out_dir, 'model.json'))
model.save_weights(join(out_dir, 'weights.json'))