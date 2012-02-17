from neuralnet import *

print 'Reading training data'
train, val, test = read_samples('data/mnistdata.csv', normalize=True, pca=150)

print 'Creating Network'
net = NeuralNet((150, 100, 10))

print 'Beginning Training'
net.train(train, val)

print 'Running Test'
net.test(test)
