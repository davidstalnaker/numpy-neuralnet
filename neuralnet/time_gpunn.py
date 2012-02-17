from neuralnet import *

def time_gpunn(name, filename, nn_structure, **kwargs):
    train, val, test = read_samples(filename, **kwargs)

    cpunn = NeuralNet(nn_structure)
    gpunn = GpuNeuralNet(nn_structure)

    print '%s: running CPU feed-forward.' % name
    t1 = timef(cpunn.test, test)
    print '%s: loading GPU feed-forward buffers.' % name
    t2 = timef(gpunn.init_feed_forward_buffers, test)
    print '%s: running GPU feed-forward.' % name
    t3 = timef(gpunn.feed_forward)

    print '%s: running CPU back-propagation.' % name
    t4 = timef(run_backprop, cpunn, train, 10000)
    print '%s: loading GPU back-propagation buffers.' % name
    t5 = timef(gpunn.init_backprop_buffers, train)
    print '%s: running GPU back-propagation.' % name
    t6 = timef(gpunn.gpu_backprop, 10000)

    return t1, t2, t3, t4, t5, t6

def run_backprop(net, samples, count):
    for i in range(count):
        net.backprop(samples[i % len(samples)])

if __name__ == '__main__':
    mnist_times = time_gpunn("MNIST", "data/mnistdata.csv",
                             (150, 100, 10), pca=150)
