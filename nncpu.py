import math
import random
import time
from numpy import mat, concatenate, vectorize, ones, multiply, power, genfromtxt
from numpy.random import rand

@vectorize
def sigmoid(x):
    if x > 25:
        return 1.0
    elif x < -25:
        return 0.0
    else:
        return 1 / (1 + math.exp(-1 * x))

@vectorize
def dsigmoid(x):
    if x > 25 or x < -25:
        return 0.0
    else:
        return math.exp(x) / (1 + math.exp(x))**2

def pad(x):
    cols = x.shape[1]
    return concatenate((mat(ones((1,cols))), x))

def readCSV(filename):
    f = file(filename)
    line = f.readline().split(',')
    if len(line) == 2:
        num_inputs = int(line[0])
        num_classes = int(line[1])
    else:
        print(line)
        raise(Exception('first line of csv should contain number of inputs, ' +
                        'number of classes'))

    samples = genfromtxt(f, delimiter=',')

    inputs = samples[:,:num_inputs]
    if samples.shape[1] == num_inputs + num_classes:
        truths = samples[:,num_inputs:]
    elif samples.shape[1] == num_inputs + 1:
        truths = map(lambda x: class_to_truth(x,num_classes), samples[:,-1])
    else:
        raise(Exception('illegal number of columns'))

    return inputs, truths

def read_and_normalize(filename, stats=None):
    inputs, truth = readCSV(filename)
    ninput, stats = normalize(inputs, stats)
    return mat_to_samplelist(ninput, truth), stats

def scale(samples, extrema=None):
    if extrema:
        mins, maxs = extrema
    else:
        maxs = samples.max(0)
        mins = samples.min(0)

    ret = (samples - mins) / (maxs - mins)
    return ret, (mins, maxs)

def normalize(samples, stats=None):
    if stats:
        means, stds = stats
    else:
        means = samples.mean(0)
        stds = samples.std(0)

    ret = (samples - means) / stds
    return ret, (means, stds)

def class_to_truth(cl, num_classes):
    truth = [0] * num_classes
    truth[int(cl)] = 1
    return truth

def truth_to_class(truth):
    return max((x,i) for i,x in enumerate(truth))[1]

def mat_to_samplelist(inputs, outputs):
    ret = []
    for i in range(inputs.shape[0]):
        ret.append((inputs[i], outputs[i]))
    return ret

def split_samples(samples, split_point=0.8):
    one = []
    two = []
    for s in samples:
        if random.random() < split_point:
            one.append(s)
        else:
            two.append(s)
    return one, two

class NeuralNet(object):
    def __init__(self, structure, eta=0.1):
        self.structure = list(structure)
        self.eta = eta
        self.reset_weights()

    def reset_weights(self):
        self.weights = []
        for i in range(len(self.structure) - 1):
            inp = self.structure[i]
            out = self.structure[i + 1]
            self.weights.append(mat(rand(inp + 1, out)) * 2 - 1)

    def backprop(self, sample):
        input = mat(sample[0]).T
        truth = mat(sample[1]).T

        sums, outputs = self.run(input.T, verbose=True)
        inputs = [input]
        inputs.extend(outputs[:-1])
        errors = []

        # calculate errors at each layer
        error = truth - outputs[-1]
        errors.insert(0, error)
        for weights in reversed(self.weights[1:]):
            error = (errors[0].T * weights.T).T[1:]
            errors.insert(0, error)

        # update each set of weights
        for i, weights in enumerate(self.weights):
            input = inputs[i]
            sum = sums[i]
            error = errors[i]

            adjustments = self.eta * pad(input) * multiply(error.T, dsigmoid(sum.T))
            self.weights[i] = self.weights[i] + adjustments

    def train(self, samples, validation, epochs=500, epoch_size=1000):
        best_rmse = 9000000001
        best_weights = []
        epochs_since_best = 0

        for num_epoch in range(epochs):

            if epochs_since_best > 50:
                break

            for s in random.sample(samples, epoch_size):
                self.backprop(s)

            misclassified, rmse = self.test(validation, to_print=False)
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = list(self.weights)
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if num_epoch % 10 == 0:
                print('Epoch %d, \tRMSE: %f' % (num_epoch, rmse))

        self.weights = best_weights

        print('\n---Final Results:---')
        print('epochs: %d' % (num_epoch - 1))
        self.test(validation)


    def run(self, input, verbose=False):
        output = mat(input).T
        sums = []
        outputs = []

        for weights in self.weights:
            sum = (pad(output).T * weights).T
            output = sigmoid(sum)
            sums.append(sum)
            outputs.append(output)

        if verbose:
            return sums, outputs
        else:
            return output

    def test(self, samples, to_print=True):
        error_sum = 0
        num_miss = 0
        for s in samples:
            input = s[0]
            truth = s[1]

            output = self.run(input)

            if truth_to_class(output) != truth_to_class(truth):
                num_miss += 1
            error = power(truth - output.T, 2).sum() / len(truth)
            error_sum += error

        misclassified = float(num_miss) / len(samples)
        rmse = math.sqrt(error_sum / len(samples))

        if to_print:
            print('misclassified: %f' % misclassified)
            print('rmse: %f' % rmse)

        return misclassified, rmse

    def time_run(self, samples, num_runs=1000):
        start = time.time()
        for i in range(num_runs):
            self.run(samples[i % len(samples)][0])
        end = time.time()

        print 'Total time: %f s' % (end - start)
        print 'Average time: %f ms per run' % ((end - start) / num_runs * 1000)

    def time_train(self, samples, num_runs=1000):
        start = time.time()
        for i in range(num_runs):
            self.backprop(samples[i % len(samples)])
        end = time.time()

        print 'Total time: %f s' % (end - start)
        print 'Average time: %f ms per run' % ((end - start) / num_runs * 1000)