from neuralnet.preprocessing import *
from neuralnet.network import *

print 'Reading training data'
train, stats = read_and_normalize('data/mnistdata.csv');

print 'Reading test data'
test, stats = read_and_normalize('data/mnisttest.csv', stats);

tmin, tmout = samplelist_to_mat(train)
tstin, tstout = samplelist_to_mat(test)
eigval, eigvec = gen_pca(tmin)

train = mat_to_samplelist(run_pca(tmin, eigval, eigvec, 150), tmout)
test = mat_to_samplelist(run_pca(tstin, eigval, eigvec, 150), tstout)

print 'Creating Network'
net = NeuralNet((150, 100, 10))

train, val = split_samples(train)

net.train(train,val)
net.test(test)
