#! /usr/bin/env python

from neuralnet.opencl import GpuNeuralNet
from neuralnet.preprocessing import utility, samplelist_to_mat
import numpy as np
from os import path
import inspect

np.set_printoptions(suppress=True)

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
gpunn.feed_forward(inputs, h_output_buf, output_buf, len(train))
gpu_output = gpunn.read_buffer(output_buf).reshape(10,2)
cpu_output = np.empty_like(gpu_output)
for i, x in enumerate(map(lambda x: gpunn.run(x).T, train)):
    cpu_output[i] = x

percent_error = 100 * abs(gpu_output - cpu_output) / ((gpu_output + cpu_output) / 2)

print('percent errors of gpu feedforward outputs:')
print(percent_error)

if np.max(percent_error) > 0.01:
    print('correct (cpu) output:')
    print(cpu_output)
    print('gpu output:')
    print(gpu_output)
else:
    print('it\'s all good')