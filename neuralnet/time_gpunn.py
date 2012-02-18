from neuralnet import *

def time_gpunn(name, filename, nn_structure, **kwargs):
    print '%s: loading data.' % name
    train, val, test = read_samples(filename, **kwargs)

    cpunn = NeuralNet(nn_structure)
    gpunn = GpuNeuralNet(nn_structure)

    print '%s: running CPU feed-forward.' % name
    cff = timef(cpunn.test, test)
    print '%s: loading GPU feed-forward buffers.' % name
    gffb = timef(gpunn.init_feed_forward_buffers, test)
    print '%s: running GPU feed-forward.' % name
    gff = timef(gpunn.feed_forward)

    print '%s: running CPU back-propagation.' % name
    cbp = timef(run_backprop, cpunn, train, 10000)
    print '%s: loading GPU back-propagation buffers.' % name
    gbpb = timef(gpunn.init_backprop_buffers, train)
    print '%s: running GPU back-propagation.' % name
    gbp = timef(gpunn.gpu_backprop, 10000)

    return cff, gffb, gff, cbp, gbpb, gbp

def run_backprop(net, samples, count):
    for i in range(count):
        net.backprop(samples[i % len(samples)])

if __name__ == '__main__':
    mnist_times = time_gpunn("MNIST", "data/mnistdata.csv",
                             (150, 100, 10), pca=150)
