from neuralnet.preprocessing import *
from neuralnet.network import *

print 'Reading training data'
train, val, test = utility('data/mnistdata.csv', 150);

print 'Creating Network'
net = NeuralNet((150, 100, 10))

print 'Beginning Training'
net.train(train,val)

print 'Running Test'
net.test(test)
