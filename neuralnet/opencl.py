import pyopencl as cl
from numpy import float32, int32, array, empty, ndarray
from neuralnet.network import NeuralNet

kernels_file = 'neuralnet/kernels.cl'
mf = cl.mem_flags

class GpuNeuralNet(NeuralNet):
    """A massively parallelized MLP implementation."""

    def __init__(self, *args, **kwargs):
        super(GpuNeuralNet, self).__init__(*args, **kwargs)
        self.ctx = cl.create_some_context()
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

    def init_buffers(self):
        """Creates OpenCL buffers."""
        self.in_weights_buf = self.make_buffer(self.weights[0])
        self.h_weights_buf = self.make_buffer(self.weights[1])
        self.h_out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * self.num_hidden)
        self.out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * self.num_output)

    def feed_forward(self, input_buf, inputOffset=0):
        self.program.feedForward(
            self.queue,
            (self.num_hidden,),
            None,
            int32(inputOffset),
            int32(self.num_input),
            int32(self.num_hidden),
            input_buf,
            self.h_out_buf,
            self.in_weights_buf)
        self.program.feedForward(
            self.queue,
            (self.num_output,),
            None,
            int32(0),
            int32(self.num_hidden),
            int32(self.num_output),
            self.h_out_buf,
            self.out_buf,
            self.h_weights_buf)
        c = empty((self.num_output,), dtype=float32)
        cl.enqueue_read_buffer(self.queue, self.out_buf, c).wait()
        print "output", c

if __name__ == "__main__":
    from neuralnet import utility, samplelist_to_mat
    train, val, test = utility('data/election/election.csv', 20)
    gpunn = GpuNeuralNet((20,40,2))
    gpunn.init_buffers()
    train_mat, train_truth = samplelist_to_mat(train)
    inputs = gpunn.make_buffer(train_mat)
    print('cpu output: %s' % gpunn.run(train[0]).T)
    gpunn.feed_forward(inputs, 0)
