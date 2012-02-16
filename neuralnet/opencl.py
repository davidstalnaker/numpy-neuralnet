#! /usr/bin/env python
import pyopencl as cl
from numpy import float32, int32, array, empty, ndarray
from network import NeuralNet

default_kernels_file = 'neuralnet/kernels.cl'
mf = cl.mem_flags

class GpuNeuralNet(NeuralNet):
    """A massively parallelized MLP implementation."""

    def __init__(self, kernels_file=default_kernels_file, *args, **kwargs):
        super(GpuNeuralNet, self).__init__(*args, **kwargs)
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.load_program(kernels_file)
        self.num_input = self.structure[0]
        self.num_hidden = self.structure[1]
        self.num_output = self.structure[2]

    def load_program(self, filename):
        """Creates an OpenCL program from the given OpenCL source file."""
        program_text = open(filename, 'r').read()
        self.program = cl.Program(self.ctx, program_text).build()

    def make_buffer(self, data, flags=mf.READ_ONLY):
        """Creates an OpenCL buffer out of the given data.

        Accepts data in the form of a NumPy array/matrix or a Python list.

        """
        if not isinstance(data, ndarray) or len(data.shape) != 1 or data.dtype != float32:
            # Convert data to a NumPy array..
            data = array(data, dtype=float32).reshape(-1)
        return cl.Buffer(self.ctx, mf.COPY_HOST_PTR | flags, hostbuf=data)

    def make_empty_buffer(self, shape, flags=mf.READ_WRITE):
        size = reduce(lambda x,y: x*y, shape)
        return cl.Buffer(self.ctx, flags, size * 4)

    def read_buffer(self, buffer):
        output = empty((buffer.size / 4,), dtype=float32)
        cl.enqueue_read_buffer(self.queue, buffer, output).wait()
        return output

    def init_buffers(self):
        """Creates OpenCL buffers."""
        self.in_weights_buf = self.make_buffer(self.weights[0])
        self.h_weights_buf = self.make_buffer(self.weights[1])
        self.h_out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * self.num_hidden)
        self.out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * self.num_output)

    def feed_forward(self, input_buf, h_out_buf, out_buf, len_input):
        self.program.feedForward(
            self.queue,
            (self.num_hidden,len_input),
            None,
            int32(self.num_input),
            int32(self.num_hidden),
            input_buf,
            h_out_buf,
            self.in_weights_buf)
        self.program.feedForward(
            self.queue,
            (self.num_output,len_input),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            h_out_buf,
            out_buf,
            self.h_weights_buf)

if __name__ == "__main__":
    from preprocessing import utility, samplelist_to_mat
    from network import timef
    from sys import argv
    filename = argv[1] if len(argv) > 1 else 'data/election/election.csv'
    kernels_file = argv[2] if len(argv) > 2 else None
    train, val, test = utility(filename, 49)
    gpunn = GpuNeuralNet(kernels_file, (49,400,2))
    gpunn.init_buffers()
    train = train[:10]
    train_mat, train_truth = samplelist_to_mat(train)
    inputs = gpunn.make_buffer(train_mat)
    h_output_buf = gpunn.make_empty_buffer((len(train), gpunn.num_hidden))
    output_buf = gpunn.make_empty_buffer((len(train), gpunn.num_output))
    timef(gpunn.feed_forward, inputs, h_output_buf, output_buf, len(train))
    for x in timef(lambda: [gpunn.run(x).T for x in train]):
        print(x)
    print(gpunn.read_buffer(output_buf).reshape(10,2))
