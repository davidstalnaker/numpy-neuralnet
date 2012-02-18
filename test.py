#! /usr/bin/env python

from neuralnet.opencl import GpuNeuralNet
from neuralnet.preprocessing import read_samples, samplelist_to_mat
import numpy as np
from os import path
import inspect
import time
try:
    import ipdb
except ImportError:
    pass

np.set_printoptions(suppress=True)

def print_backprop_info():
    
    print('correct (cpu) hidden sums:')
    print(sums[0].T)
    print('gpu hidden sums:')
    print(gpunn.read_buffer(gpunn.h_sums_buf))
    print('')
    
    print('correct (cpu) output sums:')
    print(sums[1].T)
    print('gpu output sums:')
    print(gpunn.read_buffer(gpunn.out_sums_buf))
    print('')
    
    print('correct (cpu) hidden outputs:')
    print(outputs[0].T)
    print('gpu hidden outputs:')
    print(gpunn.read_buffer(gpunn.h_out_buf))
    print('')
    
    print('correct (cpu) outputs:')
    print(outputs[1].T)
    print('gpu outputs:')
    print(gpunn.read_buffer(gpunn.out_buf))
    print('')
    
    print('correct (cpu) hidden errors:')
    print(errors[0].T)
    print('gpu hidden errors:')
    print(gpunn.read_buffer(gpunn.h_err_buf))
    print('')
    
    print('correct (cpu) output errors:')
    print(errors[1].T)
    print('gpu output errors:')
    print(gpunn.read_buffer(gpunn.out_err_buf))
    print('')
    
    print('correct (cpu) hidden weights:')
    print(deltas[0].T)
    print('gpu hidden weights:')
    iw = gpunn.read_buffer(gpunn.in_weights_buf).reshape(gpunn.num_hidden,gpunn.num_input + 1)
    print(iw)
    print('')
    
    print('correct (cpu) output weights:')
    print(deltas[1].T)
    print('')
    
    print('gpu output weights:')
    ow = gpunn.read_buffer(gpunn.h_weights_buf).reshape(gpunn.num_output,gpunn.num_hidden + 1)
    print(ow)
    print('')
    
    hidden_weights_error = np.sum(np.abs(deltas[0].T - iw))
    output_weights_error = np.sum(np.abs(deltas[1].T - ow))
    
    print('hidden weights error: %f' % hidden_weights_error)
    print('output weights error: %f' % output_weights_error)

current_dir = path.dirname(inspect.getfile(inspect.currentframe()))
data_file = path.join(current_dir, 'data/election/election.csv')
kernels_file = path.join(current_dir, 'neuralnet/kernels.cl')
train, val, test = read_samples(data_file, normalize=True, pca=2)
print('initializing...')
gpunn = GpuNeuralNet((2,3,2))
print('running...')


#start = time.time()
#gpunn.init_feed_forward_buffers(train)
#gpunn.feed_forward()
#gpu_output = gpunn.read_buffer(gpunn.ff_out_buf).reshape(len(train),2)
#end = time.time()
#print('GPU feedforward: %s ms.' % ((end - start) * 1000))
#
#cpu_output = np.empty_like(gpu_output)
#
#start = time.time()
#for i, x in enumerate(map(lambda x: gpunn.run(x).T, train)):
#    cpu_output[i] = x
#end = time.time()
#print('CPU feedforward: %s ms.' % ((end - start) * 1000))
#
#
#error = abs(gpu_output - cpu_output)
#
#print(error.shape)
#
#print('errors of gpu feedforward outputs:')
#print(error)
#
#if np.max(error) > 0.001:
#    print('correct (cpu) output:')
#    print(cpu_output)
#    print('gpu output:')
#    print(gpu_output)
#else:
#    print('feedforward is good to go')


#print('initial gpu hidden weights:')
#iw = gpunn.read_buffer(gpunn.in_weights_buf).reshape(gpunn.num_hidden,gpunn.num_input + 1)
#print(iw)
#print('')
#
#print('initial output weights:')
#ow = gpunn.read_buffer(gpunn.h_weights_buf).reshape(gpunn.num_output, gpunn.num_hidden + 1)
#print(ow)
#print('')

#deltas = gpunn.weights
#ods = []
#for i in range(30):
#    sums, outputs, errors, ds = gpunn.backprop(train[i], return_values=True)
#    ods.append(ds[1])
#    if not deltas:
#        deltas = ds
#    else:
#        deltas[0] += ds[0]
#        deltas[1] += ds[1]

gpunn.init_backprop_buffers(train, 10)
gpunn.gpu_backprop()

#print_backprop_info()

#gpunn.init_backprop_buffers(train, 1)
#gpunn.gpu_backprop(1)
#print_backprop_info()


# start = time.time()
# for i in xrange(len(train)):
#     gpunn.backpropGpu(i)
# end = time.time()
# print('GPU backprop: %s ms.' % ((end - start) * 1000))
# 
# start = time.time()
# for s in train:
#     sums, outputs, errors, weights = gpunn.backprop(s, return_values=True)
# end = time.time()
# print('CPU backprop: %s ms.' % ((end - start) * 1000))







