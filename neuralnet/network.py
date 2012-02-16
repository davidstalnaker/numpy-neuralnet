import math
import random
import time
from numpy import vectorize, multiply, power, mat, concatenate, ones
from numpy.random import rand
from .preprocessing import truth_to_class

def timef(f, *args, **kwargs):
    """Measures the runtime of the provided function and arguments."""
    start = time.time()
    r = f(*args, **kwargs)
    end = time.time()
    print('Execution time: %s ms.' % ((end - start) * 1000))
    return r

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
    """Calculates the derivative of the sigmoid of x."""
    if x > 25 or x < -25:
        return 0.0
    else:
        return math.exp(x) / (1 + math.exp(x))**2

def pad(x):
    """Adds a row of 1's to the top of a vector."""
    cols = x.shape[1]
    return concatenate((mat(ones((1,cols))), x))

class NeuralNet(object):
    def __init__(self, structure, eta=0.1):
        self.structure = list(structure)
        self.eta = eta
        self.reset_weights()

    @property
    def num_outputs(self):
        return self.structure[-1]

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

        cur_set = samples

        for num_epoch in range(epochs):

            if epochs_since_best > 50:
                break

            if len(samples) > epoch_size:
                cur_set = random.sample(samples, epoch_size)
            for s in cur_set:
                self.backprop(s)

            misclassified, rmse, _ = self.test(validation, to_print=False)
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

        # if input is a tuple, it is (input, output) - we want input only
        if type(input) == tuple:
            input = input[0]

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
        confusion = [[0] * self.num_outputs for _ in range(self.num_outputs)]
        for s in samples:
            input = s[0]
            truth = s[1]

            output = self.run(input)

            o = truth_to_class(output)
            t = truth_to_class(truth)
            confusion[o][t] += 1
            if o != t:
                num_miss += 1

            error = power(truth - output.T, 2).sum() / len(truth)
            error_sum += error

        misclassified = float(num_miss) / len(samples)
        rmse = math.sqrt(error_sum / len(samples))

        if to_print:
            print('misclassified: %f' % misclassified)
            print('rmse: %f' % rmse)
            print('confusion:')
            print(confusion)

        return misclassified, rmse, confusion

    def time(self, function, samples, num_runs=1000):
        start = time.time()
        for i in range(num_runs):
            function(samples[i % len(samples)])
        end = time.time()

        print 'Total time: %f s' % (end - start)
        print 'Average time: %f ms per run' % ((end - start) / num_runs * 1000)
