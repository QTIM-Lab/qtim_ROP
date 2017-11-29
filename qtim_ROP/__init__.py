import os  # set the keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
keras.backend.set_image_dim_ordering('tf')

import matplotlib  # set matplotlib backend
matplotlib.use('Agg')

import deep_rop
import segmentation
import preprocessing
import evaluation
import retinaunet