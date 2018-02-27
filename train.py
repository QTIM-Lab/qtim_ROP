import sys

conf = sys.argv[1]
try:
	gpu = str(sys.argv[2])
except IndexError:
	gpu = "0"

print "using gpu {}".format(gpu)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

from qtim_ROP.learning.retina_net import RetiNet
net = RetiNet(conf)
net.train()
