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
    cbp = timef(run_backprop, cpunn, train, len(train))
    print '%s: loading GPU back-propagation buffers.' % name
    gbpb = timef(gpunn.init_backprop_buffers, train)
    print '%s: running GPU back-propagation.' % name
    gbp = timef(gpunn.gpu_backprop)

    return cff, gffb, gff, cbp, gbpb, gbp

def run_backprop(net, samples, count):
    for i in range(count):
        net.backprop(samples[i % len(samples)])

def write_times(name, times):
    f = open(name + "_times.csv", "w")
    for times_tuple in times:
        f.write(",".join(map(str, times_tuple)) + "\n")

if __name__ == '__main__':
    REPS = 3
    mnist_times = [0] * REPS * 5
    poker_times = [0] * REPS * 5
    rlcp_times = [0] * REPS * 5
    election_times = [0] * REPS * 5
    for j, h in enumerate([5, 10, 50, 100, 200]):
        for i in range(REPS):
            mnist_times[j * REPS + i] = time_gpunn("MNIST",
                "data/mnisttest.csv", (150, h, 10), pca=150)
            poker_times[j * REPS + i] = time_gpunn("Poker Hands",
                "data/poker-hand/training.data", (25, h, 10))
            rlcp_times[j * REPS + i] = time_gpunn("RLCP",
                "data/rlcp/train.csv", (150, h, 10))
            election_times[j * REPS + i] = time_gpunn("Election",
                "data/election/election.csv", (150, h, 10), pca=25)
    write_times("mnist", mnist_times)
    write_times("poker", poker_times)
    write_times("rlcp", rlcp_times)
    write_times("election", election_times)
