from qtim_ROP.learning.retina_net import RetiNet
import sys

conf = sys.argv[1]

net = RetiNet(conf)
net.train()