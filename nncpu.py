import math
import csv
from numpy import mat, concatenate, vectorize, ones, multiply, power
from numpy.random import rand

training = [([1,1], [0]), ([1,0], [1]), ([0,1], [1]), ([0,0], [0])]
train2 = [
    ([0.10,0.03], [1, 0]),
    ([0.15,0.13], [1, 0]),
    ([0.07,0.05], [1, 0]),
    ([0.05,0.02], [1, 0]),
    ([0.80,0.83], [0, 1]),
    ([0.83,0.90], [0, 1]),
    ([0.90,0.91], [0, 1]),
    ([0.87,0.87], [0, 1])
]

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

def classify(input):
    return max((x,i) for i,x in enumerate(input))[1]

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

    def readCSV(self, filename):
        csvreader = csv.reader(open(filename, 'rb'), delimiter=',')
        input = []
        samples = []
        for i, row in enumerate(csvreader):
            if i % 2 == 0:
                input = map(float, row)
            else:
                samples.append((input, map(float, row)))
        return samples

    def set_normalization(self, inputs):
        self.mins = list(inputs[0])
        self.maxs = list(inputs[0])

        for input in inputs:
            for i, x in enumerate(input):
                if self.mins[i] > x:
                    self.mins[i] = x
                if self.maxs[i] < x:
                    self.maxs[i] = x

    def normalize_input(self, input):
        input = mat(input)
        return (input - self.mins) / (mat(self.maxs) - self.mins)

    def backprop(self, sample):
        input = mat(sample[0]).T
        truth = mat(sample[1]).T

        sums, outputs = self.run(input.T, verbose=True)
        inputs = [self.normalize_input(input.T).T]
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

    def train(self, samples, validation, epochs=1000):

        best_rmse = 9000000001
        best_weights = []
        epochs_since_best = 0

        for num_epoch in range(epochs):

            if epochs_since_best > 50:
                break

            for s in samples:
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
        output = self.normalize_input(mat(input)).T
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

            if classify(output) != classify(truth):
                num_miss += 1
            error = power(truth - output.T, 2).sum() / len(truth)
            error_sum += error

        misclassified = float(num_miss) / len(samples)
        rmse = math.sqrt(error_sum / len(samples))

        if to_print:
            print('misclassified: %f' % misclassified)
            print('rmse: %f' % rmse)

        return misclassified, rmse
