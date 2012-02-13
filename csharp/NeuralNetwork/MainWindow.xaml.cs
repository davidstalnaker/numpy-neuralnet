using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetwork
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public UserControl1 Graph { get; set; }

        public MainWindow()
        {
            InitializeComponent();
            Graph = new UserControl1();
            this.Content = Graph;

            var t = Generate();

        //    var s = t.Item1.Svd(true);

            //var nd = t.Item1.Multiply(s.VT().SubMatrix(0, 8, 0, 8));

            var mlp = new MLP(8, 100, 1);

            var r = mlp.Train(t.Item1, t.Item2, null, null, 2500);
            

            Graph.Set(r.TrainingSquaredError, r.TrainingError);

            DenseMatrix data = new DenseMatrix(new double[,] { { 1, 1 }, { 1, 0 }, { 0, 0 }, { 0, 1 } });
            DenseMatrix labels = new DenseMatrix(new double[,] { { 1, 0 }, { 0, 1 }, { 1, 0 }, { 0, 1 } });


            //var mlp = new MLP(2, 100, 2);

            //var r = mlp.Train(data, labels, data, labels, 5000);

            //Graph.Set(r.TrainingSquaredError, r.TrainingError);
        }

        public Tuple<DenseMatrix, DenseMatrix> Generate()
        {
            Random r = new Random();
            DenseMatrix d = new DenseMatrix((int)Math.Pow(2, 8), 8);
            DenseMatrix l = new DenseMatrix((int)Math.Pow(2, 8), 1);
            for (int i = 0; i < Math.Pow(2, 8); i++)
            {
               
                int sum = 0;
                for (int j = 0; j < 8; j++)
                {
                    var v = ((i >> j) & 0x01);


                    d[i, j] = v;

                    sum += v;
                }

                l[i,0] = sum;
            }

            return new Tuple<DenseMatrix, DenseMatrix>(d, l);

        }
    }
}
