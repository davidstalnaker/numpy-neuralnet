import random
from numpy import genfromtxt, zeros, mat, cov, dot
from numpy.linalg import eig

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

    inputs = samples[:,:num_inputs]
    if samples.shape[1] == num_inputs + num_classes:
        truths = samples[:,num_inputs:]
    elif samples.shape[1] == num_inputs + 1:
        truths = map(lambda x: class_to_truth(x,num_classes), samples[:,-1])
    else:
        raise(Exception('illegal number of columns'))

    return inputs, truths

def read_and_normalize(filename, stats=None):
    """Reads in a CSV data file and normalizes it.

    (optional) stats: (mean, variance)
    Returns: (list of samples, (mean, variance))

    """
    inputs, truth = readCSV(filename)
    ninput, stats = normalize(inputs, stats)
    return mat_to_samplelist(ninput, truth), stats


def utility(filename, num_components):
    """

    Reads samples from a file, normalizes them, performs PCA, and then
    splits them into training, validation and test sets.

    Returns a tuple of sample lists in a 60/20/20 split.

    """
    inputs, truth = readCSV(filename)
    ninputs, stats = normalize(inputs)
    vals, vecs = gen_pca(ninputs)
    print(vals)
    pca = run_pca(ninputs, vecs, num_components)

    samples = mat_to_samplelist(pca, truth)
    return split_samples(samples, (0.6,0.8))


def scale(samples, extrema=None):
    """Scales sample values into the range 0 - 1.

    samples: 2D matrix of inputs
    (optional) extrema: (mins, maxs); can be given instead of calculating.
    Returns: (scaled samples, (mins, maxs))

    """
    if extrema:
        mins, maxs = extrema
    else:
        maxs = samples.max(0)
        mins = samples.min(0)

    ret = (samples - mins) / (maxs - mins)
    return ret, (mins, maxs)


def normalize(samples, stats=None):
    """Normalizes the data into the range 0-1.

    samples: 2D matrix of inputs
    (optional) stats: (mean, variance)
    Returns: (normalized samples, (means, standard deviation))

    """
    if stats:
        means, stds = stats
    else:
        means = samples.mean(0)
        stds = samples.std(0)
    stds = map(lambda x: 1 if x == 0 else x, stds)
    ret = (samples - means) / stds
    return ret, (means, stds)


def class_to_truth(cl, num_classes):
    """Converts the class into a truth vector."""
    truth = [0] * num_classes
    truth[int(cl)] = 1
    return truth


def truth_to_class(truth):
    """Converts a truth vector into the class."""
    return max((x,i) for i,x in enumerate(truth))[1]


def mat_to_samplelist(inputs, outputs):
    """Converts a matrix back to a list of samples."""
    ret = []
    for i in range(inputs.shape[0]):
        ret.append((inputs[i], outputs[i]))
    return ret


def samplelist_to_mat(samples):
    """Converts a list of samples into a matrix.

    Returns: (inputs, outputs), where each are matrices.

    """
    first = samples[0]
    inputs = mat(zeros( (len(samples), len(first[0])) ))
    outputs = mat(zeros( (len(samples), len(first[1])) ))

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
    vecs = eigvec[:,:num_components]
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

