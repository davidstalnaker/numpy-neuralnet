using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCLTemplate;
using System.Diagnostics;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    static class Extension
    {

        public static OpenCLTemplate.CLCalc.Program.Variable ToGraphics(this float[] val)
        {
            return new CLCalc.Program.Variable(val);
        }
    }
    class OpenCLTest
    {
        string vecSum = @"
                     __kernel void
                    floatVectorSum(__global       float * v1,
                                   __global       float * v2)
                    {
                        // Vector element index
                        int i = get_global_id(0);
                        v1[i] = v1[i] + v2[i];
                    }";

        string run = @"
__kernel void runNN(__global float* source,
                    __global int* sourceLen,
                    __global float* dest,
                    __global int* destLen,
                    __global float* weights)
{

    int i = get_global_id(0);
    
    float sum = weights[i];
    
    for(int j = 0; j < (*sourceLen); j++)
    {
        sum += weights[(j + 1) * (*destLen) + i] * source[j];
    }
    
    sum = exp(sum);
    sum = 1/sum;
    sum = 1 + sum;
    sum = 1 / sum;

    dest[i] = sum;
}
";

        public OpenCLTest()
        {
            CLCalc.InitCL();
            var buildlogs = new List<string>();
            CLCalc.Program.Compile(new[] { run }, out buildlogs);
            Random r = new Random();

            float[] sample = Enumerable.Range(0, 768).Select(x => (float)r.NextDouble() * 2 - 1).ToArray();
            float[] hlW = Enumerable.Range(0, 10000 * 769).Select(x => (float)r.NextDouble() * 2 - 1).ToArray();
            float[] olW = Enumerable.Range(0, 10001 * 10).Select(x => (float)r.NextDouble() * 2 - 1).ToArray();
            float[] hlDest = new float[10000];
            float[] olDest = new float[10];

            //float[] sample = new[] { .3f, .7f };

            //float[] hlW = new[] { .4f, .4f, .1f, .5f, .5f, .7f };

            //float[] olW = new[] { .4f, .6f, .8f };

            //float[] hlDest = new[] { 0f, 0f };

            //float[] olDest = new[] { 0f };
            var sw = Stopwatch.StartNew();
            var sw2 = Stopwatch.StartNew();
            var gSample = sample.ToGraphics();
            var ghlW = hlW.ToGraphics();
            var golW = olW.ToGraphics();
            var ghlDest = hlDest.ToGraphics();
            var golDest = olDest.ToGraphics();
            sw2.Stop();

            var kernel = new CLCalc.Program.Kernel("runNN");

            var args = new[] {gSample, new CLCalc.Program.Variable(new[] { sample.Length}), ghlDest,
               new CLCalc.Program.Variable(new[] { hlDest.Length}), ghlW};

            kernel.Execute(args, hlDest.Length);





            args = new CLCalc.Program.Variable[5];
            args[0] = ghlDest;
            args[1] = new CLCalc.Program.Variable(new[] { hlDest.Length });
            args[2] = golDest;
            args[3] = new CLCalc.Program.Variable(new[] { olDest.Length });
            args[4] = golW;

            kernel.Execute(args, olDest.Length);

            sw2.Start();
            ghlDest.ReadFromDeviceTo(hlDest);
            golDest.ReadFromDeviceTo(olDest);
            sw2.Stop();
            sw.Stop();

            Console.WriteLine("hlDest\n" + string.Join("\n", hlDest));
            Console.WriteLine("olDest\n" + string.Join("\n", olDest));

            Console.WriteLine("Total {0}ms", sw.ElapsedMilliseconds);
            Console.WriteLine("Memory {0}ms", sw2.ElapsedMilliseconds);
        }


        //    public OpenCLTest()
        //    {
        //        CLCalc.InitCL();

        //        CLCalc.Program.Compile(new[] { vecSum });

        //        var kernel = new CLCalc.Program.Kernel("floatVectorSum");

        //        int n = 30000000;

        //        float[] v1 = new float[n];
        //        float[] v2 = new float[n];

        //        for (int i = 0; i < n; i++)
        //        {
        //            v1[i] = (float)i;
        //            v2[i] = 2.0f;
        //        }
        //        var sw = Stopwatch.StartNew();
        //        var sw2 = Stopwatch.StartNew();

        //        OpenCLTemplate.CLCalc.Program.Variable varV1 = new CLCalc.Program.Variable(v1);
        //        OpenCLTemplate.CLCalc.Program.Variable varV2 = new CLCalc.Program.Variable(v2);
        //        sw2.Stop();
        //        OpenCLTemplate.CLCalc.Program.Variable[] args = new CLCalc.Program.Variable[] { varV1, varV2 };

        //        int[] workers = new int[1] { n };

        //        //Console.WriteLine(string.Join("\n", v1));

        //        kernel.Execute(args, workers);

        //        sw2.Start();
        //        varV1.ReadFromDeviceTo(v1);
        //        sw2.Stop();
        //        //Console.WriteLine(string.Join("\n", v1));
        //        sw.Stop();

        //        for (int i = 0; i < 100; i++)
        //        {
        //            Console.WriteLine("{0} + {1} = {2}", i, 2.0f, v1[i]);
        //        }

        //        Console.WriteLine("Total {0}ms", sw.ElapsedMilliseconds);
        //        Console.WriteLine("Memory {0}ms", sw2.ElapsedMilliseconds);

        //        CPU(v1, v2);
        //    }

        //    public void CPU(float[] v1, float[] v2)
        //    {
        //        var sw = Stopwatch.StartNew();
        //        Parallel.For(0, v1.Length, (i) =>
        //        {
        //            v1[i] += v2[i];
        //        });
        //        sw.Stop();
        //        Console.WriteLine("CPU {0}ms", sw.ElapsedMilliseconds);
        //    }
        //}
    }
}
