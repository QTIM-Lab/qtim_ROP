import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
keras.backend.set_image_dim_ordering('th')

import deep_rop
import segmentation