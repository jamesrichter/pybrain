print "mucking around with pybrain"
print "building a network..."

import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
import datetime
print datetime.datetime.now()

net = buildNetwork(2, 3, 1)

net.activate([2,1])
net['in']
net['hidden0']
net['out']

net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer)
net['hidden0']
net = buildNetwork(2, 3, 2, hiddenclass=TanhLayer, outclass = SoftmaxLayer)
net = buildNetwork(2,3,1, bias=True)
net['bias']

print datetime.datetime.now()