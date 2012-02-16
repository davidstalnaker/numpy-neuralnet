#! /usr/bin/env python

from neuralnet.opencl import GpuNeuralNet
from neuralnet.preprocessing import utility, samplelist_to_mat
from neuralnet.network import timef
from os import path
import inspect

current_dir = path.dirname(inspect.getfile(inspect.currentframe()))
data_file = path.join(current_dir, 'data/election/election.csv')
kernels_file = path.join(current_dir, 'neuralnet/kernels.cl')
train, val, test = utility(data_file, 49)
print('initializing...')
gpunn = GpuNeuralNet(kernels_file, (49,400,2))
print('running...')
gpunn.init_buffers()
train = train[:10]
train_mat, train_truth = samplelist_to_mat(train)
inputs = gpunn.make_buffer(train_mat)
h_output_buf = gpunn.make_empty_buffer((len(train), gpunn.num_hidden))
output_buf = gpunn.make_empty_buffer((len(train), gpunn.num_output))
timef(gpunn.feed_forward, inputs, h_output_buf, output_buf, len(train))
for x in timef(lambda: [gpunn.run(x).T for x in train]):
    print(x)
print(gpunn.read_buffer(output_buf).reshape(10,2))
