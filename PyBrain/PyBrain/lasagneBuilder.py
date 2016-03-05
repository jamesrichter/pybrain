#To use lasagne, theano is required; to use theano, CUDA is required, to install CUDA, Visual Studio 2013 /or less/ is required.  COMPUTERS
print "could not get CUDA/theano working :("
#import lasagne
#import theano
#import theano.tensor as T

#input_var = T.tensor4('X')
#target_var = T.ivector('y')

#from lasagne.nonlinearities import leaky_rectify, softmax

#network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var)
#network = lasagne.layers.Conv2DLayer(network, 64, (3, 3),
#                                     nonlinearity=leaky_rectify)
#network = lasagne.layers.Conv2DLayer(network, 32, (3, 3),
#                                     nonlinearity=leaky_rectify)
#network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')
#network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
#                                    128, nonlinearity=leaky_rectify,
#                                    W=lasagne.init.Orthogonal())
#network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    #10, nonlinearity=softmax)

