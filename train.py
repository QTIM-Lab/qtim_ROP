import sys
from glob import glob
from qtim_ROP.__main__ import initialize
conf_dict, conf_file = initialize()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = conf_dict['gpu']

from qtim_ROP.learning.retina_net import RetiNet
confs = glob(sys.argv[1])
out_dir = sys.argv[2] if len(sys.argv) > 2 else None

for c in confs:
    print(c)
    net = RetiNet(c, out_dir=out_dir)
    net.train()
