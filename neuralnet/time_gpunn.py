from neuralnet import *

def time_gpunn(name, csv_file, pca_num, nn_structure):
    train, val, test = read_samples(csv_file, pca=pca_num);
    cpunn = NeuralNet(nn_structure)

    print '%s running sequential training.' % name
    timef(cpunn.train, train, val, epochs=100)

    print '%s running sequential test.' % name
    timef(cpunn.test, test)
    gpunn = GpuNeuralNet(nn_structure)

    print '%s running sequential training.' % name
    timef(gpunn.train, train, val, epochs=100)

    print '%s running sequential test.' % name
    timef(gpunn.test, test)
    
def run_backprop(count, net, samples):
    for i in range(count):
        net.backprop(samples[i % len(samples)])

if __name__ == '__main__':
    time_gpunn("MNIST", "data/mnistdata.csv", 150, (150, 100, 10))
