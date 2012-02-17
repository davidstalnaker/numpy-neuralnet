#! /usr/bin/env python

from neuralnet.opencl import GpuNeuralNet
from neuralnet.preprocessing import read_samples, samplelist_to_mat
import numpy as np
from os import path
import inspect
import time
np.set_printoptions(suppress=True)

current_dir = path.dirname(inspect.getfile(inspect.currentframe()))
data_file = path.join(current_dir, 'data/election/election.csv')
kernels_file = path.join(current_dir, 'neuralnet/kernels.cl')
train, val, test = read_samples(data_file, normalize=True, pca=2)
print('initializing...')
gpunn = GpuNeuralNet((2,3,2))
print('running...')

train_mat, train_truth = samplelist_to_mat(train)

start = time.time()
gpunn.init_buffers()
inputs = gpunn.make_buffer(train_mat)
h_output_buf = gpunn.make_empty_buffer((len(train), gpunn.num_hidden))
output_buf = gpunn.make_empty_buffer((len(train), gpunn.num_output))

gpunn.feed_forward(inputs, h_output_buf, output_buf, len(train))
gpu_output = gpunn.read_buffer(output_buf).reshape(len(train),2)
end = time.time()
print('GPU feedforward: %s ms.' % ((end - start) * 1000))

cpu_output = np.empty_like(gpu_output)

start = time.time()
for i, x in enumerate(map(lambda x: gpunn.run(x).T, train)):
    cpu_output[i] = x
end = time.time()
print('CPU feedforward: %s ms.' % ((end - start) * 1000))


error = abs(gpu_output - cpu_output)

print(error.shape)

print('errors of gpu feedforward outputs:')
print(error)

if np.max(error) > 0.001:
    print('correct (cpu) output:')
    print(cpu_output)
    print('gpu output:')
    print(gpu_output)
else:
    print('feedforward is good to go')

#sample = train[0]
#sums, outputs, errors, weights = gpunn.backprop(sample, return_values=True)

#start = time.time()
#for s in train:
#    gpunn.backpropGpu(s[0], s[1])
#end = time.time()
#print('GPU backprop: %s ms.' % ((end - start) * 1000))
#
#start = time.time()
#for s in train:
#    gpunn.backprop(s)
#end = time.time()
#print('CPU backprop: %s ms.' % ((end - start) * 1000))

#print('correct (cpu) hidden sums:')
#print(sums[0].T)
#print('correct (cpu) output sums:')
#print(sums[1].T)
#print('correct (cpu) hidden outputs:')
#print(outputs[0].T)
#print('correct (cpu) outputs:')
#print(outputs[1].T)
#print('correct (cpu) hidden errors:')
#print(errors[0])
#print('correct (cpu) output errors:')
#print(errors[1])
#print('correct (cpu) hidden weights:')
#print(weights[0].T)
#print('correct (cpu) output weights:')
#print(weights[1].T)
#print('\n')

#print('gpu hidden sums:')
#print(gpunn.read_buffer(h_sums_buf))
#print('gpu output sums:')
#print(gpunn.read_buffer(out_sums_buf))
#print('gpu hidden outputs:')
#print(gpunn.read_buffer(h_out_buf))
#print('gpu outputs:')
#print(gpunn.read_buffer(out_buf))
#print('gpu hidden errors:')
#print(gpunn.read_buffer(h_err_buf))
#print('gpu output errors:')
#print(gpunn.read_buffer(out_err_buf))
#print('gpu hidden weights:')
#iw = gpunn.read_buffer(gpunn.in_weights_buf).reshape(gpunn.num_hidden,gpunn.num_input + 1)
#print(iw)
#print('gpu output weights:')
#ow = gpunn.read_buffer(gpunn.h_weights_buf).reshape(gpunn.num_output, gpunn.num_hidden + 1)
#print(ow)

#hidden_weights_error = np.sum(np.abs(weights[0].T - iw))
#output_weights_error = np.sum(np.abs(weights[1].T - ow))

#print('hidden weights error: %f' % hidden_weights_error)
#print('output weights error: %f' % output_weights_error)




