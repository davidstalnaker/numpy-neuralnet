import random
from numpy import genfromtxt, zeros, mat, cov
from numpy.linalg import eig

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

def samplelist_to_mat(samples):
    first = samples[0]
    inputs = mat(zeros( (len(samples), len(first[0])) ))
    outputs = mat(zeros( (len(samples), len(first[1])) ))

    for i, s in enumerate(samples):
        inputs[i] = s[0]
        outputs[i] = s[1]

    return inputs, outputs

def gen_pca(matrix):
    c = cov(matrix.T)
    eigval, eigvec = eig(c)
    return eigval, eigvec

def run_pca(matrix, eigval, eigvec, num_components):
    vecs = eigvec[:,:num_components]
    return matrix * vecs

def split_samples(samples, split_point=0.8):
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