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

namespace NeuralNetwork
{
    /// <summary>
    /// Interaction logic for UserControl1.xaml
    /// </summary>
    public partial class UserControl1 : UserControl
    {
        public UserControl1()
        {
            InitializeComponent();
        }

        public void Set(List<double> squared, List<double> error)
        {
            _triangleSeries.ItemsSource = squared.Select((x, i) => new KeyValuePair<double, double>(x, i*10)).ToArray();
            _lineSeries.ItemsSource = error.Select((x, i) => new KeyValuePair<double, double>(x, i*10)).ToArray();
        }
    }
}
