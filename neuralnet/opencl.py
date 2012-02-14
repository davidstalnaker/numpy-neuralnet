import pyopencl as cl
from numpy import float32, int32, array, empty

class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def popCorn(self):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays
        self.num_input = 2
        self.num_hidden = 2
        self.num_output = 1
        self.input = array([.3, .7], dtype=float32)
        self.in_weights = array([.4, .4, .1, .5, .5, .7], dtype=float32)
        self.h_weights = array([.4, .6, .8], dtype=float32)

        #create OpenCL buffers
        self.input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input)
        self.in_weights_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.in_weights)
        self.h_weights_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_weights)
        self.h_out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * 2)
        self.out_buf = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * 1)

    def execute(self):
        self.program.run(self.queue, (self.num_input,), None, int32(self.num_input), int32(self.num_hidden), self.input_buf, self.h_out_buf, self.in_weights_buf)
        self.program.run(self.queue, (self.num_hidden,), None, int32(self.num_hidden), int32(self.num_output), self.h_out_buf, self.out_buf, self.h_weights_buf)
        c = empty((self.num_output,), dtype=float32)
        cl.enqueue_read_buffer(self.queue, self.out_buf, c).wait()
        print "input", self.input
        print "output", c



if __name__ == "__main__":
    example = CL()
    example.loadProgram("kernels.cl")
    example.popCorn()
    example.execute()