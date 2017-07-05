import os  # set the keras backend
os.environ['KERAS_BACKEND'] = 'theano'
import keras
keras.backend.set_image_dim_ordering('th')

import matplotlib  # set matplotlib backend
matplotlib.use('Agg')

import deep_rop
import segmentation