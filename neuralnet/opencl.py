import pyopencl as cl
from numpy import float32, int32, array, empty
from neuralnet.network import NeuralNet

kernels_file = 'neuralnet/kernels.cl'
mf = cl.mem_flags

class GpuNeuralNet(NeuralNet):
    def __init__(self, structure, eta=0.1):
        super(GpuNeuralNet, self).__init__(structure, eta)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.load_program(kernels_file)

    def load_program(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def init_mem_weights(self):

        #initialize client side (CPU) arrays
        self.num_input = self.structure[0]
        self.num_hidden = self.structure[1]
        self.num_output = self.structure[2]
        self.input = array(input, dtype=float32)
        self.in_weights = array(self.weights[0].reshape(-1), dtype=float32)
        self.h_weights = array(self.weights[1].reshape(-1), dtype=float32)

        #create OpenCL buffers
        self.input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input)
        self.in_weights_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.in_weights)
        self.h_weights_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_weights)
        self.h_out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * self.num_hidden)
        self.out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * self.num_output)

    def execute(self):
        self.program.run(self.queue, (self.num_hidden,), None, int32(self.num_input), int32(self.num_hidden), self.input_buf, self.h_out_buf, self.in_weights_buf)
        self.program.run(self.queue, (self.num_output,), None, int32(self.num_hidden), int32(self.num_output), self.h_out_buf, self.out_buf, self.h_weights_buf)
#        c = empty((self.num_output,), dtype=float32)
#        cl.enqueue_read_buffer(self.queue, self.out_buf, c).wait()
#        print "input", self.input
#        print "output", c



if __name__ == "__main__":
    example = GpuNeuralNet((2,4,2))
    example.init_mem([.1,.9])
    example.execute()