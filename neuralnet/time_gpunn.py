from neuralnet import *

def time_gpunn(name, filename, nn_structure, **kwargs):
    train, val, test = read_samples(filename, **kwargs)
    cpunn = NeuralNet(nn_structure)

    print '%s running sequential training.' % name
    t1 = timef(cpunn.train, train, val, epochs=100)

    print '%s running sequential test.' % name
    t2 = timef(cpunn.test, test)
    gpunn = GpuNeuralNet(nn_structure)

    print '%s running sequential training.' % name
    t3 = timef(gpunn.train, train, val, epochs=100)

    print '%s running sequential test.' % name
    t4 = timef(gpunn.test, test)
    return t1, t2, t3, t4

def run_backprop(count, net, samples):
    for i in range(count):
        net.backprop(samples[i % len(samples)])

if __name__ == '__main__':
    mnist_times = time_gpunn("MNIST", "data/mnistdata.csv",
                             (150, 100, 10), pca=150)
