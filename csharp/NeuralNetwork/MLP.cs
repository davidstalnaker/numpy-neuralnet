using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Generic;

namespace NeuralNetwork
{
    public class MLP
    {
        public double LearningRate { get; set; }
        public int Features { get; private set; }
        public int HiddenNodes { get; private set; }
        public int Classes { get; private set; }

        private Matrix<double> hiddenLayer;
        private Matrix<double> outputLayer;


        public MLP(int features, int hiddenNodes, int classes)
        {
            LearningRate = .1;
            Features = features;
            HiddenNodes = hiddenNodes;
            Classes = classes;

            hiddenLayer = RandMatrix(features, hiddenNodes);
            outputLayer = RandMatrix(hiddenNodes + 1, classes);
        }

        private Random _rnd = new Random();

        private Matrix<double> RandMatrix(int rows, int cols)
        {
            var ret = new DenseMatrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    ret[i, j] = _rnd.NextDouble() * 2 - 1;
                }
            }

            return ret;
        }

        public TrainingResults Train(Matrix<double> trainingData, Matrix<double> trainingClasses, Matrix<double> validation, Matrix<double> validationClasses, int epochs = 500)
        {
            var ret = new TrainingResults();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var newWeights = CalcNewWeights(trainingData, trainingClasses);
                hiddenLayer = newWeights.Item1;
                outputLayer = newWeights.Item2;
                if (epoch % 10 == 0)
                {
                    double s, p;

                    CalcSquaredError(trainingData, trainingClasses, out s, out p);
                    ret.TrainingSquaredError.Add(s);
                    ret.TrainingError.Add(p);

                    if (validation != null && validationClasses != null)
                    {
                        CalcSquaredError(validation, validationClasses, out s, out p);
                        ret.ValidationSquaredError.Add(s);
                        ret.ValidationError.Add(p);
                    }

                    if (epoch % 100 == 0)
                        Console.WriteLine("Epoch {0}, Error {1}, {2}", epoch, s, p * 100);
                }
            }

            return ret;
        }

        private void CalcSquaredError(Matrix<double> trainingData, Matrix<double> trainingClasses, out double squaredError, out double percentError)
        {
            squaredError = 0;
            percentError = 0;
            for (int i = 0; i < trainingData.RowCount; i++)
            {
                var a = Classify(trainingData.Row(i));
                var expected = trainingClasses.Row(i);

                squaredError += (a.Subtract(expected).Select(x => x * x).Sum());

                if (expected[a.MaximumIndex()] == 0)
                    percentError++;
            }

            percentError /= trainingData.RowCount;
        }


        private Tuple<Matrix<double>, Matrix<double>> CalcNewWeights(Matrix<double> trainingData, Matrix<double> trainingClasses)
        {
            Matrix<double> localHL = hiddenLayer.Clone();
            Matrix<double> localOL = outputLayer.Clone();
            for (int i = 0; i < trainingData.RowCount; i++)
            {
                var sample = trainingData.Row(i);
                var target = trainingClasses.Row(i);
                UpdateWeightsIndividual(ref localHL, ref localOL, sample, target);


            }

            return new Tuple<Matrix<double>, Matrix<double>>(localHL, localOL);

        }

        public Vector<double> Classify(Vector<double> sample)
        {
            var hlOut = Sigmoid(hiddenLayer.LeftMultiply(sample));

            hlOut = Prepend(1, hlOut);

            var v = Sigmoid(outputLayer.LeftMultiply(hlOut));
           
            return v;
        }

        private void UpdateWeightsIndividual(ref Matrix<double> localHL, ref Matrix<double> localOL, Vector<double> sample, Vector<double> target)
        {
            var hlOut = Sigmoid(localHL.LeftMultiply(sample));

            hlOut = Prepend(1, hlOut);

            var v = Sigmoid(localOL.LeftMultiply(hlOut));
            
            var deltaOL = v.Subtract(target).PointwiseMultiply(v).PointwiseMultiply(OneMinusV(v));

            var deltaHL = localOL.Transpose().LeftMultiply(deltaOL).PointwiseMultiply(hlOut).PointwiseMultiply(OneMinusV(hlOut));

            var tmp = deltaOL.Multiply(LearningRate).ToColumnMatrix();

            localOL = localOL.Subtract(tmp.Multiply(hlOut.ToRowMatrix()).Transpose());

            tmp = deltaHL.SubVector(1, deltaHL.Count - 1).ToColumnMatrix();
            tmp = tmp.Multiply(LearningRate);

            localHL = localHL.Subtract(tmp.Multiply(sample.ToRowMatrix()).Transpose());
        }

        private Vector<double> OneMinusV(Vector<double> v)
        {
            var ret = v.CreateVector(v.Count);
            for (int i = 0; i < v.Count; i++)
            {
                ret[i] = 1 - v[i];
            }
            return ret;
        }

        private Vector<double> Prepend(double val, Vector<double> v)
        {
            var ret = v.CreateVector(v.Count + 1);
            ret[0] = val;

            for (int i = 0; i < v.Count; i++)
            {
                ret[i + 1] = v[i];
            }
            return ret;
        }


        private Vector<double> Sigmoid(Vector<double> v)
        {
            var ret = v.CreateVector(v.Count);
            for (int i = 0; i < v.Count; i++)
            {
                var a = Math.Exp(v[i]);

                a = 1 / a;
                a = 1 + a;
                a = 1 / a;
                ret[i] = a;
            }

            return ret;
        }

        public class TrainingResults
        {
            public List<double> ValidationSquaredError { get; set; }
            public List<double> ValidationError { get; set; }
            public List<double> TrainingSquaredError { get; set; }
            public List<double> TrainingError { get; set; }

            public TrainingResults()
            {
                ValidationSquaredError = new List<double>();
                ValidationError = new List<double>();
                TrainingError = new List<double>();
                TrainingSquaredError = new List<double>();
            }
        }
    }
}
