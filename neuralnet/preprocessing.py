"""Defines various helper functions for working with data sets.

Terms:

"data":    a matrix of _input_ data
"truth":   a truth matrix (_output_ data)
"samples": a Python list of (input, output) data pairs

"""
import random

from numpy import cov, dot, genfromtxt, mat, zeros
from numpy.linalg import eig

def read_samples(filenames, normalize=False, scale=False,
                 pca=None, split=(0.6, 0.8)):
    """Reads samples from file(s), optionally doing various transformations.

    For each transformation, the information (stats/extrema/eigvecs)
    from the first file's data is used for every successive data set.

    filenames: the file to read from, or a tuple of files.
    (optional) normalize: bool, whether to normalize.
    (optional) scale:     bool, whether to scale.
    (optional) pca:       the number of PCA components to select.
    (optional) split:     if a single file is given, how to split it.

    Returns a tuple of sample lists.  If only one file was given,
    a 60/20/20 split is used by default.

    """
    if isinstance(filenames, basestring):
        filenames = [filenames]
    data_sets, truths = zip(*map(read_csv, filenames))
    data_sets = list(data_sets)
    if normalize:
        stats = None
        for i, data in enumerate(data_sets):
            data, stats = normalize_data(data, stats)
            data_sets[i] = data
    elif scale:
        extrema = None
        for i, data in enumerate(data_sets):
            data, extrema = scale_data(data, extrema)
            data_sets[i] = data
    if pca:
        _, eigvecs = gen_pca(data_sets[0])
        data_sets = [run_pca(data, eigvecs, pca) for data in data_sets]
    if len(data_sets) > 1:
        return tuple(zip(data, truth) for data, truth
                                      in zip(data_sets, truths))
    else:
        return split_samples(zip(data_sets[0], truths[0]), (0.6, 0.8))

def read_csv(filename):
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
