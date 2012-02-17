"""Defines various helper functions for working with data sets.

Terms:

"data":    a matrix of _input_ data
"truth":   a truth matrix (_output_ data)
"samples": a Python list of (input, output) data pairs

"""

import random
from numpy import genfromtxt, zeros, mat, cov, dot
from numpy.linalg import eig

def read_samples(filename, normalize=False, scale=False, pca=None):
    """Reads samples from a file, optionally doing various transformations.

    filename:  the file to read from.
    normalize: bool, whether to normalize.
    scale:     bool, whether to scale.
    pca:       the number of PCA components to select.
    Returns a tuple of sample lists in a 60/20/20 split.

    """
    data, truth = readCSV(filename)
    if normalize:
        data, _ = normalize_data(data)
    elif scale:
        data, _ = scale_data(data)
    if pca:
        vals, vecs = gen_pca(data)
        data = run_pca(data, vecs, pca)
    return split_samples(zip(data, truth), (0.6, 0.8))

def readCSV(filename):
    """Parses a CSV file.

    Returns: ([data inputs], [truth vectors])

    """
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

    inputs = samples[:, :num_inputs]
    if samples.shape[1] == num_inputs + num_classes:
        truths = samples[:, num_inputs:]
    elif samples.shape[1] == num_inputs + 1:
        truths = map(lambda x: class_to_truth(x, num_classes), samples[:, -1])
    else:
        raise(Exception('illegal number of columns'))

    return inputs, truths

def scale_data(data, extrema=None):
    """Scales data values into the range 0 - 1.

    data: 2D matrix of inputs
    (optional) extrema: (mins, maxs); can be given instead of calculating.
    Returns: (scaled samples, (mins, maxs))

    """
    if extrema:
        mins, maxs = extrema
    else:
        maxs = data.max(0)
        mins = data.min(0)

    ret = (data - mins) / (maxs - mins)
    return ret, (mins, maxs)

def normalize_data(data, stats=None):
    """Normalizes the data into the range 0-1.

    data: 2D matrix of inputs
    (optional) stats: (mean, variance)
    Returns: (normalized samples, (means, standard deviation))

    """
    if stats:
        means, stds = stats
    else:
        means = data.mean(0)
        stds = data.std(0)
    stds = map(lambda x: 1 if x == 0 else x, stds)
    ret = (data - means) / stds
    return ret, (means, stds)

def class_to_truth(cls_num, num_classes):
    """Converts the class into a truth vector."""
    truth = zeros((num_classes,))
    truth[int(cls_num)] = 1
    return truth

def truth_to_class(truth):
    """Converts a truth vector into the class."""
    return max((x,i) for i,x in enumerate(truth))[1]

def samplelist_to_mat(samples):
    """Converts a list of samples into a matrix.

    Returns: (inputs, outputs), where each are matrices.

    """
    first = samples[0]
    inputs =  mat(zeros((len(samples), len(first[0]))))
    outputs = mat(zeros((len(samples), len(first[1]))))

    for i, s in enumerate(samples):
        inputs[i] = s[0]
        outputs[i] = s[1]

    return inputs, outputs

def gen_pca(matrix):
    """Returns (eigenvalues, eigenvectors) of the given matrix."""
    c = cov(matrix.T)
    eigval, eigvec = eig(c)
    return eigval, eigvec

def run_pca(matrix, eigvec, num_components):
    """Performs PCA on the given matrix of data."""
    vecs = eigvec[:, :num_components]
    return dot(matrix, vecs)

def split_samples(samples, split_point=0.8):
    """Splits sample data probabilistically into multiple chunks.

    split_point: a tuple of points to split the data at.
        For example, (0.6, 0.8):
            ~60% to the first group
            ~20% (0.8 - 0.6) to the second group
            ~20% (1.0 - 0.8) to the third group
    Returns a tuple of sample lists.

    """
    if type(split_point) == tuple:
        num_splits = len(split_point)
    else:
        num_splits = 1
        split_point = (split_point,)

    split = tuple([[] for x in range(num_splits + 1)])

    for s in samples:
        r = random.random()
        for i, p in enumerate(split_point):
            if r < p:
                split[i].append(s)
                break
        else:
            split[-1].append(s)
    return split
