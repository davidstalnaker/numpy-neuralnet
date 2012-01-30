import math
import random
import time
from numpy import vectorize, multiply, power, mat, concatenate, ones
from numpy.random import rand
from neuralnet.preprocessing import truth_to_class

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