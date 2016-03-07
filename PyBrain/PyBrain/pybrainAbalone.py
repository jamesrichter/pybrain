from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import sklearn as sk
import numpy as np
from sklearn.cross_validation import train_test_split

array = np.loadtxt('abalone.txt', dtype = "U", delimiter = ',')
array[array == "M"] = "-1"
array[array == "F"] = "1"
array[array == "I"] = "0"
house_data = array[:,0:-1]
house_target = array[:,-1]

alldata = ClassificationDataSet(8, 1, nb_classes=40)
for n in xrange(house_data.__len__()):
    datapt = np.array([float(house_data[n][x]) for x in range(house_data[n].__len__())])
    target = [float(house_target[n])]
    alldata.addSample(datapt, target)

tstdata, trndata = alldata.splitWithProportion( 0.25 )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0]
print trndata['target'][0]
print trndata['class'][0]

fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

#ticks = arange(-3.,6.,0.2)
#X, Y = meshgrid(ticks, ticks)
## need column vectors in dataset, not arrays
#griddata = ClassificationDataSet(8,1, nb_classes=40)
#for i in xrange(X.size):
#    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
#griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

for i in range(20):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

    #out = fnn.activateOnDataset(griddata)
    #out = out.argmax(axis=1)  # the highest output activation gives the class
    #out = out.reshape(X.shape)

    #figure(1)
    #ioff()  # interactive graphics off
    #clf()   # clear the plot
    #hold(True) # overplot on
    #for c in [0,1,2]:
    #    here, _ = where(tstdata['class']==c)
    #    plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    #if out.max()!=out.min():  # safety check against flat field
    #    contourf(X, Y, out)   # plot the contour
    #ion()   # interactive graphics on
    #draw()  # update the plot

#ioff()
#show()
