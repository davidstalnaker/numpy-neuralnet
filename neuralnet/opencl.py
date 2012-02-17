import inspect
from os import path

import pyopencl as cl
from numpy import float32, int32, array, empty, ndarray

from .network import NeuralNet

current_dir = path.dirname(inspect.getfile(inspect.currentframe()))
kernels_file = path.join(current_dir, 'kernels.cl')

mf = cl.mem_flags

class GpuNeuralNet(NeuralNet):
    """A massively parallelized MLP implementation."""

    def __init__(self, structure, learning_rate=0.1):
        assert len(structure) == 3, "Must have 3 layers."
        super(GpuNeuralNet, self).__init__(structure, learning_rate=0.1)
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
        size = reduce(lambda x, y: x * y, shape)
        return cl.Buffer(self.ctx, flags, size * 4)

    def read_buffer(self, buffer):
        output = empty((buffer.size / 4,), dtype=float32)
        cl.enqueue_read_buffer(self.queue, buffer, output).wait()
        return output

    def init_buffers(self):
        """Creates OpenCL buffers."""
        self.in_weights_buf = self.make_buffer(self.weights[0].T)
        self.h_weights_buf = self.make_buffer(self.weights[1].T)
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
            (self.num_output, len_input),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            h_out_buf,
            out_buf,
            self.h_weights_buf)

    def backpropGpu(self, input, truth):

        in_buf =       self.make_buffer(input)
        truth_buf =    self.make_buffer(truth)
        h_sums_buf =   self.make_empty_buffer((self.num_hidden, 1))
        h_out_buf =    self.make_empty_buffer((self.num_hidden, 1))
        out_sums_buf = self.make_empty_buffer((self.num_output, 1))
        out_buf =      self.make_empty_buffer((self.num_output, 1))
        h_err_buf =    self.make_empty_buffer((self.num_hidden, 1))
        out_err_buf =  self.make_empty_buffer((self.num_output, 1))

        self.program.feedForwardTraining(
            self.queue,
            (self.num_hidden, 1),
            None,
            int32(self.num_input),
            int32(self.num_hidden),
            in_buf,
            h_out_buf,
            h_sums_buf,
            self.in_weights_buf
        )

        self.program.feedForwardTraining(
            self.queue,
            (self.num_output, 1),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            h_out_buf,
            out_buf,
            out_sums_buf,
            self.h_weights_buf
        )

        self.program.outputError(
            self.queue,
            (self.num_output, 1),
            None,
            out_buf,
            truth_buf,
            out_err_buf
        )

        self.program.hiddenError(
            self.queue,
            (self.num_hidden, 1),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            out_err_buf,
            self.h_weights_buf,
            h_err_buf
        )

        self.program.updateWeights(
            self.queue,
            (self.num_hidden, 1),
            None,
            int32(self.num_input),
            int32(self.num_hidden),
            int32(self.lr),
            h_err_buf,
            self.in_weights_buf,
            h_sums_buf
        )

        self.program.updateWeights(
            self.queue,
            (self.num_output, 1),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            int32(self.lr),
            out_err_buf,
            self.h_weights_buf,
            out_sums_buf
        )

        return h_sums_buf, h_out_buf, out_sums_buf, out_buf, h_err_buf, out_err_buf
