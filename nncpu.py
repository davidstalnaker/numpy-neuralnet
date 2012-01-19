import sys
import math
from numpy import mat, concatenate, vectorize, ones, multiply
from numpy.random import rand

training = [([1,1], [0]), ([1,-1], [1]), ([-1,1], [1]), ([-1,-1], [0])]
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
        self.input_nodes = structure[0]
        self.hidden_nodes = structure[1]
        self.output_nodes = structure[2]
        self.eta = eta
        self.reset_weights()

    def reset_weights(self):
        self.hidden_weights = mat(rand(self.input_nodes + 1, self.hidden_nodes)) * 2 - 1
        self.output_weights = mat(rand(self.hidden_nodes + 1, self.output_nodes)) * 2 - 1

        #print(self.hidden_weights)
        #print(self.output_weights)

    def train(self, samples, epochs=1000):
        w_1 = self.hidden_weights
        w_2 = self.output_weights

        for i in range(epochs):
            for s in samples:
                x = mat(s[0]).T
                y = mat(s[1]).T
                a, b = self.run(x.T, verbose=True)
                a_1 = a[0]
                a_2 = a[1]
                b_1 = b[0]
                b_2 = b[1]

                d_2 = y - b_2
                print('error = %f' % d_2[0][0])
                d_1 = (d_2.T * w_2.T).T

                e_1 = self.eta * pad(x) * multiply(d_1[1:].T, dsigmoid(a_1.T))
                self.hidden_weights = self.hidden_weights + e_1

                e_2 = self.eta * b_1 * multiply(d_2.T, dsigmoid(a_2.T))
                self.output_weights = self.output_weights + e_2


#        except ValueError as e:
#            print(e.message)
#            tb = sys.exc_traceback
#            while tb.tb_next:
#                tb = tb.tb_next
#            args = tb.tb_frame.f_locals
#            print('first:')
#            print(args['self'])
#            print('second:')
#            print(args['other'])




    def run(self, input, verbose=False):
        w_1 = self.hidden_weights
        w_2 = self.output_weights
        x = pad(mat(input).T)

        a_1 = (x.T * w_1).T
        b_1 = pad(sigmoid(a_1))

        a_2 = (b_1.T * w_2).T
        b_2 = sigmoid(a_2)

        if verbose:
            return [a_1,a_2], [b_1,b_2]
        else:
            return b_2


if __name__ == '__main__':
    net = NeuralNet((2,3,1))
    net.train()
    print(net.run([1,-1]))
