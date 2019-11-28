/*
This file implements a simple neural network class and wrapppers to access it fom Python. 
We use the mean square error as loss function and a sigmoid as activation function. 
(We use that, if f is the sigmoid, f' = (1-f)*f.)
We assume that
	* the number of neurons in the first layer is equal to the number of inputs,
	* there is exactly one neuron in the last layer.
*/

#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <vector>
#include <boost/python.hpp>

using namespace std;
using namespace boost::python;

template <typename T>
vector<T> python_list_to_vector(list& l, T x){ // second argument used for template deduction
	vector<T> res;
	for(int i=0; i < len(l); i++){
		res.push_back(extract<T>(l[i]));
	}
	return res;
}

template <typename T>
list vector_to_python_list(vector<T> v){
	list res;
	for(int i=0; i<v.size(); i++){
		res.append(v[i]);
	}
	return res;
}

template <typename T>
vector<vector<T>> python_list_of_lists_to_vector(list& l, T x){ // second argument used for template deduction
	vector<vector<T>> res;
	for(int i=0; i < len(l); i++){
		vector<T> line;
		for(int j=0; j < len(l[i]); j++){
			line.push_back(extract<T>(l[i][j]));
		}
		res.push_back(line);
	}
	return res;
}

double sigmoid(double x){
	return (1. / (1. + exp(-x)));
}

class NeuralNetwork1{

	private:

		int N_layers;
		vector<int> layers; 
		vector<vector<vector<double>>> weights;
		vector<vector<double>> bias;

	public:
		
		// constructor with random weights and bias

		void build_network(vector<int> layers_){
			N_layers = layers_.size();
			default_random_engine generator(time(0));
			normal_distribution<double> distribution(0.,1.);
			int N_k;
			for(int i=0; i<N_layers; i++){
				layers.push_back(layers_[i]);
				vector<vector<double>> weights_layer;
				vector<double> bias_layer;
				for(int j=0; j<layers[i]; j++){
					vector<double> weights_neuron;
					if(i > 0){
						N_k = layers[i-1];
					}
					else{
						N_k = layers[0]; // assume the first layer as as many neurons as there are inputs
					}
					for(int k=0; k<N_k; k++){
						weights_neuron.push_back(distribution(generator));
					}
					weights_layer.push_back(weights_neuron);
					bias_layer.push_back(distribution(generator));
				}
				weights.push_back(weights_layer);
				bias.push_back(bias_layer);
			}
		}

		NeuralNetwork1(vector<int> layers_){
			build_network(layers);
		}
		
		// constructor with random weights and bias - python case
		NeuralNetwork1(list& layers_l){
			build_network(python_list_to_vector(layers_l, 0));
		}
		
		// constructor with given weights and bias
		NeuralNetwork1(vector<int> layers_, 
			          vector<vector<vector<double>>> weights_, 
		              vector<vector<double>> bias_){
			N_layers = layers_.size();
			layers = layers_;
			weights = weights_;
			bias = bias_;
		}

		// constructor accepting Python lists instead of vectors
		NeuralNetwork1(list& layers_, 
			          list& weights_, 
		              list& bias_){
			N_layers = len(layers_);
			for(int i=0; i<N_layers; i++){
				layers.push_back(extract<int>(layers_[i]));
				vector<vector<double>> weights_layer;
				vector<double> bias_layer;
				for(int j=0; j<layers[i]; j++){
					vector<double> weights_neuron;
					for(int k=0; k<len(weights_[i][j]); k++){
						weights_neuron.push_back(extract<double>(weights_[i][j][k]));
					}
					weights_layer.push_back(weights_neuron);
					bias_layer.push_back(extract<double>(bias_[i][j]));
				}
				weights.push_back(weights_layer);
				bias.push_back(bias_layer);
			}
		}

		vector<double> feedforward(vector<double> x){
			for(int i=0; i<N_layers; i++){
				vector<double> y;
				for(int j=0; j<layers[i]; j++){
					double z = bias[i][j];
					for(int k=0; k<x.size(); k++){
						z += weights[i][j][k]*x[k];
					}
					y.push_back(sigmoid(z));
				}
				x = y;
			}
			return x;
		}
	
		// feedforward using a Python list as input
		list feedforward_python(list& x){
			vector<double> y = feedforward(python_list_to_vector(x, 0.));
			return vector_to_python_list(y);
		}

		// evaluating the loss function
		double loss(vector<vector<double>> data, vector<double> y_true_all){
			double res = 0.;
			for(int i=0; i<y_true_all.size(); i++){
				res += pow(feedforward(data[i])[0] - y_true_all[i], 2);
			}
			return res / y_true_all.size();
		}
		
		// evaluating the loss function - Python
		double loss_python(list& data, list& y_true_all){
			return loss(python_list_of_lists_to_vector(data ,0.), python_list_to_vector(y_true_all, 0.));
		}
		
		// training function
		void train(vector<vector<double>> data, vector<double> y_true_all, double learn_rate, long epochs){
			for(long epoch = 0; epoch < epochs; epoch++){
				for(int index_data = 0; index_data < y_true_all.size(); index_data++){
					vector<double> x = data[index_data];
					double y_true = y_true_all[index_data];
					
					// feedforward, retaining the state of each neuron
					vector<vector<double>> states;
					states.push_back(x);
					for(int i=0; i<N_layers; i++){
						vector<double> y;
						for(int j=0; j<layers[i]; j++){
							double z = bias[i][j];
							for(int k=0; k<x.size(); k++){
								z += weights[i][j][k]*x[k];
							}
							y.push_back(sigmoid(z));
						}
						states.push_back(y);
						x = y;
					}

					//predicted value: state of the last neuron (the last layer is assumed to contain only one neuron)
					double y_pred = states[N_layers][0]; 

					// derivative of the loss function with respect to y_pred
					double d_L_d_ypred = 2.*(y_pred - y_true);

					// partial derivatives
					vector<vector<vector<double>>> d_ypred_d_x;
					vector<vector<vector<double>>> d_ypred_d_weights;
					vector<vector<double>> d_ypred_d_bias;

					//partial derivatives - output layer
					double state = states[N_layers][0];
					vector<vector<double>> d_ypred_d_x_layer;
					vector<double> d_ypred_d_x_neuron;
					vector<vector<double>> d_ypred_d_weights_layer;
					vector<double> d_ypred_d_weights_neuron;
					vector<double> d_ypred_d_bias_layer;
					for(int k=0; k<layers[N_layers-2]; k++){
						d_ypred_d_x_neuron.push_back(weights[N_layers-1][0][k]*state*(1.-state));
						d_ypred_d_weights_neuron.push_back(states[N_layers-1][k]*state*(1.-state));
					}
					d_ypred_d_x_layer.push_back(d_ypred_d_x_neuron);
					d_ypred_d_weights_layer.push_back(d_ypred_d_weights_neuron);
					d_ypred_d_bias_layer.push_back(state*(1.-state));

					d_ypred_d_x.insert(d_ypred_d_x.begin(), d_ypred_d_x_layer);
					d_ypred_d_weights.insert(d_ypred_d_weights.begin(), d_ypred_d_weights_layer);
					d_ypred_d_bias.insert(d_ypred_d_bias.begin(), d_ypred_d_bias_layer);
					
					//partial derivatives - other layers
					for(int i=2; i<N_layers+1; i++){
						vector<vector<double>> d_ypred_d_x_layer;
						vector<vector<double>> d_ypred_d_weights_layer;
						vector<double> d_ypred_d_bias_layer;
						for(int j=0; j<layers[N_layers - i]; j++){
							double d_ypred_d_yint = 0.;
							for(int k=0; k<layers[N_layers-i+1]; k++){
								d_ypred_d_yint += d_ypred_d_x[0][k][j];
							}
							double state = states[N_layers-i+1][j];
							vector<double> d_ypred_d_x_neuron;
							vector<double> d_ypred_d_weights_neuron;
							for(int k=0; k<weights[N_layers-i][j].size(); k++){
								d_ypred_d_x_neuron.push_back(weights[N_layers-i][j][k]*state*(1.-state)*d_ypred_d_yint);
								d_ypred_d_weights_neuron.push_back(states[N_layers-i][k]*state*(1.-state)*d_ypred_d_yint);
							}
							d_ypred_d_x_layer.push_back(d_ypred_d_x_neuron);
							d_ypred_d_weights_layer.push_back(d_ypred_d_weights_neuron);
							d_ypred_d_bias_layer.push_back(state*(1.-state)*d_ypred_d_yint);
						}
						d_ypred_d_x.insert(d_ypred_d_x.begin(), d_ypred_d_x_layer);
						d_ypred_d_weights.insert(d_ypred_d_weights.begin(), d_ypred_d_weights_layer);
						d_ypred_d_bias.insert(d_ypred_d_bias.begin(), d_ypred_d_bias_layer);
					}

					// update weights and bias
					for(int i=0; i<N_layers; i++){
						for(int j=0; j<layers[i]; j++){
							for(int k=0; k<weights[i][j].size(); k++){
								weights[i][j][k] -= learn_rate * d_L_d_ypred * d_ypred_d_weights[i][j][k];
							}
							bias[i][j] -= learn_rate * d_L_d_ypred * d_ypred_d_bias[i][j];
						}
					}
				}
			}
		}
		
		// training function - Python
		void train_python(list& data, list& y_true_all, double learn_rate, long epochs){
			train(python_list_of_lists_to_vector(data ,0.), python_list_to_vector(y_true_all, 0.), learn_rate, epochs);
		}

		// save the neural network to a file
		void save(char* filename){
			ofstream file;
			file.open(filename);
			file << N_layers << '\n';
			for(int i=0; i<N_layers; i++){
				file << layers[i] << '\t';
			}
			file << '\n';
			for(int i=0; i<N_layers; i++){
				for(int j=0; j<layers[i]; j++){
					file << weights[i][j].size() << '\t';
					for(int k=0; k<weights[i][j].size(); k++){
						file << weights[i][j][k] << '\t';
					}
					file << bias[i][j] << '\t';
				}
			}
			file.close();
		}
		
		// load the neural network from a file
		void load(char* filename){
			ifstream file;
			file.open(filename);
			file >> N_layers;
			layers.clear();
			long n_neurons;
			long n_weights;
			double weight_;
			double bias_;
			for(int i=0; i<N_layers; i++){
				file >> n_neurons;
				layers.push_back(n_neurons);
			}
			weights.clear();
			bias.clear();
			for(int i=0; i<N_layers; i++){
				vector<vector<double>> weights_layer;
				vector<double> bias_layer;
				for(int j=0; j<layers[i]; j++){
					vector<double> weights_neuron;
					file >> n_weights;
					for(int k=0; k<n_weights; k++){
						file >> weight_;
						weights_neuron.push_back(weight_);
					}
					file >> bias_;
					bias_layer.push_back(bias_);
					weights_layer.push_back(weights_neuron);
				}
				weights.push_back(weights_layer);
				bias.push_back(bias_layer);
			}
			file.close();
		}

		// implement load
};


BOOST_PYTHON_MODULE(NN1)
{
    class_<NeuralNetwork1>("NeuralNetwork1", init<list&>())
		.def(init<list&, list&, list&>())
		.def("feedforward", &NeuralNetwork1::feedforward_python)
		.def("loss", &NeuralNetwork1::loss_python)
		.def("train", &NeuralNetwork1::train_python)
		.def("save", &NeuralNetwork1::save)
		.def("load", &NeuralNetwork1::load)
	;
}
