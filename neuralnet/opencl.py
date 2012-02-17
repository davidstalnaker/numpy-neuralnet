import inspect
from os import path

import pyopencl as cl
from numpy import float32, int32, array, empty, ndarray

from .network import NeuralNet
from .preprocessing import samplelist_to_mat

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
        self.init_buffers()

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

    def init_backprop_buffers(self, train_samples):
        self.train_inputs, self.train_truth = samplelist_to_mat(train_samples)
        self.train_inputs = self.train_inputs.reshape(-1)
        self.train_truth = self.train_truth.reshape(-1)
        
        self.num_samples = len(train_samples)
        
        self.in_buf =       self.make_buffer(self.train_inputs)
        self.truth_buf =    self.make_buffer(self.train_truth)
        self.h_sums_buf =   self.make_empty_buffer((self.num_hidden, 1))
        self.h_out_buf =    self.make_empty_buffer((self.num_hidden, 1))
        self.out_sums_buf = self.make_empty_buffer((self.num_output, 1))
        self.out_buf =      self.make_empty_buffer((self.num_output, 1))
        self.h_err_buf =    self.make_empty_buffer((self.num_hidden, 1))
        self.out_err_buf =  self.make_empty_buffer((self.num_output, 1))
        

    def backpropGpu(self, sample_num):
        input = self.in_buf.get_sub_region(4 * self.num_input * sample_num, 4 * self.num_input)
        truth = self.truth_buf.get_sub_region(4 * self.num_output * sample_num, 4 * self.num_output)
        
        self.program.feedForwardTraining(
            self.queue,
            (self.num_hidden, 1),
            None,
            int32(self.num_input),
            int32(self.num_hidden),
            input,
            self.h_out_buf,
            self.h_sums_buf,
            self.in_weights_buf
        )

        self.program.feedForwardTraining(
            self.queue,
            (self.num_output, 1),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            self.h_out_buf,
            self.out_buf,
            self.out_sums_buf,
            self.h_weights_buf
        )

        self.program.outputError(
            self.queue,
            (self.num_output, 1),
            None,
            self.out_buf,
            truth,
            self.out_err_buf
        )

        self.program.hiddenError(
            self.queue,
            (self.num_hidden, 1),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            self.out_err_buf,
            self.h_weights_buf,
            self.h_err_buf
        )

        self.program.updateWeights(
            self.queue,
            (self.num_hidden, 1),
            None,
            int32(self.num_input),
            int32(self.num_hidden),
            float32(self.lr),
            input,
            self.h_err_buf,
            self.in_weights_buf,
            self.h_sums_buf
        )

        self.program.updateWeights(
            self.queue,
            (self.num_output, 1),
            None,
            int32(self.num_hidden),
            int32(self.num_output),
            float32(self.lr),
            self.h_out_buf,
            self.out_err_buf,
            self.h_weights_buf,
            self.out_sums_buf
        )
