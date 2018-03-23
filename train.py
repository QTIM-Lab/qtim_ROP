import sys
from glob import glob


try:
    gpu = str(sys.argv[2])
except IndexError:
    gpu = "0"

print("using gpu {}".format(gpu))

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

from qtim_ROP.learning.retina_net import RetiNet
confs = glob(sys.argv[1])

for c in confs:
    net = RetiNet(c)
    net.train()
    del net