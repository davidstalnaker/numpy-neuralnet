import math
from numpy import mat, concatenate, vectorize, ones, multiply
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
    return (1 / (1 + math.exp(-1 * x)))

@vectorize
def dsigmoid(x):
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

    def train(self, samples, epochs=1000):
        for i in range(epochs):
            for s in samples:
                input = mat(s[0]).T
                truth = mat(s[1]).T

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

                print(error)

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


if __name__ == '__main__':
    net = NeuralNet((2,3,1))
    net.train()
    print(net.run([1,-1]))
