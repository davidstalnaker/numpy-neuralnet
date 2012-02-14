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

__kernel void run(int numInputs,
                  int numOutputs,
                  __global float* input,
                  __global float* output,
                  __global float* weights)
{
    int i = get_global_id(0);
    
    float sum = weights[i];
    
    for (int j = 0; j < numInputs; j++)
    {
        sum += weights[(j + 1) * numOutputs + i] * input[j];
    }

    output[i] = sigmoid(sum);
}

__kernel void runForTraining(int numInputs,
                               int numOutputs,
                               __global float* input,
                               __global float* output,
                               __global float* sums,
                               __global float* weights)
{
    int i = get_global_id(0);
    
    float sum = weights[i];
    
    for (int j = 0; j < numInputs; j++)
    {
        sum += weights[(j + 1) * numOutputs + i] * input[j];
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