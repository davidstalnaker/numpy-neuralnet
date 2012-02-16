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
    int i = get_global_id(0);
    int inputOffset = get_global_id(1) * inputSize;
    int outputOffset = get_global_id(1) * outputSize;
    
    float sum = weights[i];
    
    for (int in = 0; in < inputSize; in++)
    {
        sum += weights[(in + 1) * outputSize + i] * inputs[in + inputOffset];
    }

    outputs[outputOffset + i] = sigmoid(sum);
}

__kernel void runForTraining(int inputSize,
                               int outputSize,
                               __global float* input,
                               __global float* output,
                               __global float* sums,
                               __global float* weights)
{
    int i = get_global_id(0);
    
    float sum = weights[i];
    
    for (int j = 0; j < inputSize; j++)
    {
        sum += weights[(j + 1) * outputSize + i] * input[j];
    }
	sums[i] = sum;
    output[i] = sigmoid(sum);
}

__kernel void outputError(__global float* output,
                          __global float* truth,
                          __global float* error)
{
    int i = get_global_id(0);

	error[i] = truth[i] - output[i];
}

__kernel void hiddenError(int numHidden,
                          int numOutputs,
                          __global float* outputError,
                          __global float* weights,
                          __global float* hiddenError)
{
    int h = get_global_id(0);

	for (int o = 0; o < numOutputs; o++) 
	{
		hiddenError[h] += weights[(h + 1) * numHidden + o] * outputError[o];
	}
}

__kernel void updateWeights(int numInputs,
                            int numOutputs,
                            int eta,
                            __global float* errors,
                            __global float* weights,
                            __global float* sums)
{
	int o = get_global_id(0);

	for (int i = 0; i < numInputs + 1; i++)
	{
		weights[i * numOutputs + o] += eta * errors[o] * dsigmoid(sums[o]);
	}
}