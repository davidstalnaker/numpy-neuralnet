float sigmoid(float x);
float dsigmoid(float x);

float sigmoid(float x)
{
    x = exp(x);
    x = 1 / x;
    x = 1 + x;
    return 1 / x;
}

float dsigmoid(float x)
{
    float e = exp(x);
    return e/((1+e)*(1+e));
}

__kernel void feedForward(int inputSize,
                          int outputSize,
                          __global float* inputs,
                          __global float* outputs,
                          __global float* weights)
{
    int on = get_global_id(0);
    int inputOffset = get_global_id(1) * inputSize;
    int outputOffset = get_global_id(1) * outputSize;
    
    float sum = weights[on * (inputSize + 1)];
    
    for (int in = 0; in < inputSize; in++)
    {
        sum += weights[(on * (inputSize + 1)) + in + 1] * inputs[in + inputOffset];
    }

    outputs[outputOffset + on] = sigmoid(sum);
}

__kernel void feedForwardTraining(int inputSize,
                                  int outputSize,
                                  int offset,
                                  __global float* inputs,
                                  __global float* outputs,
                                  __global float* sums,
                                  __global float* weights)
{
    int on = get_global_id(0);
    int inputOffset = (offset + get_global_id(1)) * inputSize;
    int outputOffset = get_global_id(1) * outputSize;

    float sum = weights[on * (inputSize + 1)];

    for (int in = 0; in < inputSize; in++)
    {
        sum += weights[(on * (inputSize + 1)) + in + 1] * inputs[in + inputOffset];
    }

    sums[outputOffset + on] = weights[0];
    outputs[outputOffset + on] = sigmoid(sum);
}

__kernel void outputError(int outputSize,
                          int errorOffset,
                          __global float* output,
                          __global float* truth,
                          __global float* error)
{
    
    int offset = (errorOffset + get_global_id(1)) * outputSize;
    int i = offset + get_global_id(0);

    error[i] = truth[i] - output[i];
}

__kernel void hiddenError(int numHidden,
                          int numOutputs,
                          __global float* outputError,
                          __global float* weights,
                          __global float* hiddenError)
{
    int h = get_global_id(0);
    int hiddenOffset = numHidden * get_global_id(1);
    int outputOffset = numOutputs * get_global_id(1);

    hiddenError[hiddenOffset + h] = 0;

    for (int o = 0; o < numOutputs; o++)
    {
        hiddenError[hiddenOffset + h] += weights[(o * (numHidden + 1)) + h + 1] * outputError[outputOffset + o];
    }
}

__kernel void updateDeltas(int numInputs,
                           int numOutputs,
                           float lr,
                           int offset,
                           __global float* inputs,
                           __global float* errors,
                           __global float* weights,
                           __global float* sums)
{
    int o = get_global_id(0);
    int inputOffset = numInputs * (offset + get_global_id(1));
    int outputOffset = numOutputs * get_global_id(1);
    int weightOffset = (numInputs + 1) * numOutputs * get_global_id(1);
    
    weights[weightOffset + (o * (numInputs + 1))] = lr * errors[outputOffset + o] * dsigmoid(sums[outputOffset + o]);
    for (int i = 0; i < numInputs; i++)
    {
        weights[weightOffset + (o * (numInputs + 1)) + i + 1] = lr * inputs[inputOffset + i] * errors[outputOffset + o] * dsigmoid(sums[outputOffset + o]);
    }
}

__kernel void updateWeights(int numInputs,
                            int numOutputs,
                            int blockSize,
                            __global float* deltas,
                            __global float* weights)
{
    int i = get_global_id(0);
    int o = get_global_id(1);
    int deltaOffset = (numInputs + 1) * numOutputs;
    
    float delta = 0.0;
    for (int n = 0; n < blockSize; n++) {
        delta += deltas[(n * deltaOffset) + (o * (numInputs + 1)) + i];
    }
    weights[(o * (numInputs + 1)) + i] += delta;
}
                            

