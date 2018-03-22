import matplotlib  # set matplotlib backend
matplotlib.use('Agg')

import os  # set the keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
keras.backend.set_image_dim_ordering('tf')

from . import deep_rop
from . import segmentation
from . import preprocessing
from . import evaluation
from . import retinaunet