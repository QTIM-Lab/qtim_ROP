import matplotlib  # set matplotlib backend
matplotlib.use('Agg')


from .__main__ import initialize
conf_dict, _ = initialize()


import os  # set the keras backend
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['KERAS_BACKEND'] = 'tensorflow'

if conf_dict['gpu'] is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = conf_dict['gpu']

import keras
keras.backend.set_image_dim_ordering('tf')

from . import deep_rop
from . import segmentation
from . import preprocessing
from . import evaluation
from . import retinaunet
from  . import quality_assurance