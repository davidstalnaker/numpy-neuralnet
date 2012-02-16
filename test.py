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
train, val, test = utility(data_file, 2)
print('initializing...')
gpunn = GpuNeuralNet(kernels_file, (2,3,2))
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

error = abs(gpu_output - cpu_output)

print('errors of gpu feedforward outputs:')
print(error)

if np.max(error) > 0.001:
    print('correct (cpu) output:')
    print(cpu_output)
    print('gpu output:')
    print(gpu_output)
else:
    print('feedforward is good to go')

sample = train[0]
sums, outputs, errors, weights = gpunn.backprop(sample, return_values=True)

#print('correct (cpu) hidden sums:')
#print(sums[0].T)
#print('correct (cpu) output sums:')
#print(sums[1].T)
#print('correct (cpu) hidden outputs:')
#print(outputs[0].T)
#print('correct (cpu) outputs:')
#print(outputs[1].T)
print('correct (cpu) hidden errors:')
print(errors[0])
print('correct (cpu) output errors:')
print(errors[1])
#print('correct (cpu) hidden weights:')
#print(weights[0].T)
#print('correct (cpu) output weights:')
#print(weights[1].T)
print('\n')

h_sums_buf, h_out_buf, out_sums_buf, out_buf, h_err_buf, out_err_buf = gpunn.backpropGpu(sample[0], sample[1])

#print('gpu hidden sums:')
#print(gpunn.read_buffer(h_sums_buf))
#print('gpu output sums:')
#print(gpunn.read_buffer(out_sums_buf))
#print('gpu hidden outputs:')
#print(gpunn.read_buffer(h_out_buf))
#print('gpu outputs:')
#print(gpunn.read_buffer(out_buf))
print('gpu hidden errors:')
print(gpunn.read_buffer(h_err_buf))
print('gpu output errors:')
print(gpunn.read_buffer(out_err_buf))





