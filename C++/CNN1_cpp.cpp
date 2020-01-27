#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>  
#include <random>
#include <thread>  
#include <boost/multi_array.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

#include "CNN1_layers_cpp.cpp"

// precision for floating numbers written in files
constexpr int prec_write_file = 16; 

// define a standard normal distribution
std::random_device rd;  
std::mt19937 gen(rd()); 
std::normal_distribution<> dis(0.,1.);

struct double_and_int{
	double d = 0.;
	int i = 0;
};

class CNN1{

	private: 

		int num_CLs; // number of convolution layers
		int img_h_i; // image height
		int img_w_i; // image width
		int num_images; // number of images
		int num_labels; // number of different possible labels
		
		vector<int> CL_size_filters; // Convolution layers: filter sizes
		vector<int> CL_num_filters; // Convolution layers: numbers of filters
		vector<int> MP_size; // Maxpool layers: pool sizes
		
		// These four vectors will contain the layers
		vector<ConvLayer> CLs; // Convolution layers
		vector<ReLU> RLUs; // ReLU layers
		vector<MaxPool> MPs; // Maxpool layers
		vector<SoftMax> SMs; // Softmax layers

	public: 

		CNN1(){
			np::initialize(); // required to create numpy arrays (otherwise leads to segmentation faults)
		}	

		CNN1(
			int img_w_i_, // image width 
			int img_h_i_, // image height
			p::list& CL_size_filters_, // list of filter sizes
			p::list& CL_num_filters_, // list of numbers of filters
			p::list& MP_size_, // list of pool sizes
			int num_labels_ // number of diffeent possible labels
		) 
		{
			// initialization
			img_w_i = img_w_i_;
			img_h_i = img_h_i_;
			num_labels = num_labels_;
			CL_size_filters = int_list_to_vector(CL_size_filters_);
			CL_num_filters = int_list_to_vector(CL_num_filters_);
			MP_size = int_list_to_vector(MP_size_);
			num_CLs = len(CL_size_filters_);
			
			num_images = 1; // tracks the number of images
			int img_w = img_w_i; // tracks the width of images 
			int img_h = img_h_i; // tracks the height of images

			// build the layers
			for(int i=0; i<num_CLs; i++){
				CLs.push_back(ConvLayer(CL_size_filters[i], CL_num_filters[i], gen, dis));
				num_images = num_images * CL_num_filters[i];
				img_w = img_w + 1 - CL_size_filters[i];
				img_h = img_h + 1 - CL_size_filters[i];
				RLUs.push_back(ReLU());
				MPs.push_back(MaxPool(MP_size[i]));
				img_w = (int) img_w / MP_size[i];
				img_h = (int) img_h / MP_size[i];
			}
			SMs.push_back(SoftMax(num_images*img_h*img_w, num_labels, gen, dis));

			np::initialize(); // required to create numpy arrays (otherwise leads to segmentation faults)
		}

		// save the CNN parameters to a file
		void save(char* filename){
			ofstream file;
			file.open(filename);
			file << fixed << setprecision(prec_write_file);
			file << num_CLs << sep_val << img_w_i << sep_val << img_h_i << sep_val << num_images << sep_line;
			save_vector(CL_size_filters, file);
			save_vector(CL_num_filters, file);
			save_vector(MP_size, file); 
			for(int i=0; i<num_CLs; i++){
				CLs[i].save(file);
				RLUs[i].save(file);
				MPs[i].save(file);
			}
			SMs[0].save(file);
			file.close();
		}
		
		// load the CNN parameters from a file
		void load(char* filename){
			ifstream file;
			file.open(filename);
			char c;
			file >> num_CLs >> c >> img_w_i >> c >> img_h_i >> c >> num_images >> c;
			CL_size_filters = load_vector<int>(file);
			CL_num_filters = load_vector<int>(file);
			MP_size = load_vector<int>(file); 
			CLs.clear();
			RLUs.clear();
			CLs.clear();
			SMs.clear();
			for(int i=0; i<num_CLs; i++){
				CLs.push_back(ConvLayer());
				CLs[i].load(file);
				RLUs.push_back(ReLU());
				RLUs[i].load(file);
				MPs.push_back(MaxPool());
				MPs[i].load(file);
			}
			SMs.push_back(SoftMax());
			SMs[0].load(file);
			file.close();
		}

		// forward pass
		vector<double> forward(d3_array_type &input){
			
			// number and dimensions of images
			int nim = input.shape()[0];
			int h = input.shape()[1];
			int w = input.shape()[2];
			if(h != img_h_i || w != img_w_i){
				cout << "\nInput dimension not divisible by the pool size—information will be lost!\n" << endl;
			}
			for(int i=0; i<num_CLs; i++){
				d3_array_type output1 = CLs[i].forward(input);
				input.resize(boost::extents[output1.shape()[0]][output1.shape()[1]][output1.shape()[2]]);
				input = output1;

				d3_array_type output2 = RLUs[i].forward(input);
				input.resize(boost::extents[output2.shape()[0]][output2.shape()[1]][output2.shape()[2]]);
				input = output2;

				d3_array_type output3 = MPs[i].forward(input);
				input.resize(boost::extents[output3.shape()[0]][output3.shape()[1]][output3.shape()[2]]);
				input = output3;
			}
			return SMs[0].forward(input);
		}
		
		// backpropagation
		void backprop(vector<double> d_L_d_out_i, double learn_rate){
			
			auto shape = SMs[0].last_input.shape();
			d3_array_type d_L_d_in(boost::extents[shape[0]][shape[1]][shape[2]]); 
		
			d_L_d_in = SMs[0].backprop(d_L_d_out_i, learn_rate);
			
			for(int i=num_CLs-1; i>=0; i--){
				d3_array_type d_L_d_in3 = MPs[i].backprop(d_L_d_in);
				shape = d_L_d_in3.shape();
				d_L_d_in.resize(boost::extents[shape[0]][shape[1]][shape[2]]);
				d_L_d_in = d_L_d_in3;
			
				d3_array_type d_L_d_in2 = RLUs[i].backprop(d_L_d_in);
				shape = d_L_d_in2.shape();
				d_L_d_in.resize(boost::extents[shape[0]][shape[1]][shape[2]]);
				d_L_d_in = d_L_d_in2;
				
				d3_array_type d_L_d_in1 = CLs[i].backprop(d_L_d_in, learn_rate);
				shape = d_L_d_in1.shape();
				d_L_d_in.resize(boost::extents[shape[0]][shape[1]][shape[2]]);
				d_L_d_in = d_L_d_in1;
			}
		}
		
		// loss function and accuracy (1 if correct answer, 0 otherwise)
		double_and_int loss_acc(vector<double> &output, int label){
			double_and_int results;
			results.d = -log(output[label]); 
			results.i = 1;
			for(int i=0; i<output.size(); i++){
				if(output[i] > output[label]){
					results.i = 0;
				}
			}
			return results;
		}

		// Completes a training step on the image 'image' with label 'label'.
		// Returns the corss-entropy and accuracy.
		double_and_int train(d2_array_type image, int label, double learn_rate = 0.005) {
			
			int h = image.shape()[0];
			int w = image.shape()[1];

			// conversion to a 3d array
			d3_array_type images(boost::extents[1][h][w]);
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					images[0][i][j] = image[i][j];
				}
			}
			
			// forward pass
			vector<double> output_forward = forward(images);
	
			// gradient of the loss function with respect to the output
			vector<double> d_L_d_out;
			for(int i=0; i<num_labels; i++){
				d_L_d_out.push_back(0.);
			}
			d_L_d_out[label] = -1./output_forward[label];

			// backpropagation
			backprop(d_L_d_out, learn_rate);

			// return loss and accuracy
			return loss_acc(output_forward, label);
		}
		
		// test function: forward propagation for one convolution layer
		np::ndarray test_forward_CL(np::ndarray image){
			return CLs[0].test_forward(image);
		}
		
		// test function: backpropagation for one convolution layer
		np::ndarray test_backprop_CL(np::ndarray image, double learn_rate){
			return CLs[0].test_backprop(image, learn_rate);
		}
		
		// test function: forward propagation for one ReLU layer
		np::ndarray test_forward_RLU(np::ndarray image){
			return RLUs[0].test_forward(image);
		}
		
		// test function: backpropagation for one ReLU layer
		np::ndarray test_backprop_RLU(np::ndarray image){
			return RLUs[0].test_forward(image);
		}
		
		// test function: forward propagation for one MaxPool layer
		np::ndarray test_forward_MP(np::ndarray image){
			return MPs[0].test_forward(image);
		}
		
		// test function: backpropagation for one MaxPool layer
		np::ndarray test_backprop_MP(np::ndarray image){
			return MPs[0].test_backprop(image);
		}
		
		// test function: forward propagation for the Softmax layer
		np::ndarray test_forward_SM(np::ndarray image){
			return SMs[0].test_forward(image);
		}
		
		// test function: backpropagation for the Softmax layer
		np::ndarray test_backprop_SM(np::ndarray d_L_d_out, double learn_rate){
			return SMs[0].test_backprop(d_L_d_out, learn_rate);
		}

		// full forward propagation - Python wrapper
		// input: 2d numpy array
		np::ndarray forward_python(np::ndarray image){
			d2_array_type input_p = d2_numpy_to_multi_array(image);
			auto shape = input_p.shape();
			d3_array_type input(boost::extents[1][shape[0]][shape[1]]);
			for(int i=0; i<shape[0]; i++){
				for(int j=0; j<shape[1]; j++){
					input[0][i][j] = input_p[i][j];
				}
			}
			return vector_to_numpy(forward(input));
		}

		// forward - return loss and accuracy - Python wrapper
		p::list forward_la_python(np::ndarray image, int label){
			d2_array_type input_p = d2_numpy_to_multi_array(image);
			auto shape = input_p.shape();
			d3_array_type input(boost::extents[1][shape[0]][shape[1]]);
			for(int i=0; i<shape[0]; i++){
				for(int j=0; j<shape[1]; j++){
					input[0][i][j] = input_p[i][j];
				}
			}
			vector<double> output = forward(input);
			double_and_int results = loss_acc(output, label);
			p::list results_p;
			results_p.append(results.d);
			results_p.append(results.i);
			return results_p;
		}
		
		// full backpropagation - Python wrapper
		void backprop_python(np::ndarray d_L_d_out, double learn_rate){
			backprop(numpy_to_vector(d_L_d_out), learn_rate);
		}
		
		// train - Python wrapper
		p::list train_python(np::ndarray image, int label, double learn_rate) {
			double_and_int results;
			results = train(d2_numpy_to_multi_array(image), label, learn_rate);
			p::list results_p;
			results_p.append(results.d);
			results_p.append(results.i);
			return results_p;
		}

};

BOOST_PYTHON_MODULE(CNN1_cpp)
{
    p::class_<CNN1>("CNN1", p::init<int, int, p::list&, p::list&, p::list&, int>())
		.def(p::init<>())
		.def("test_forward_CL", &CNN1::test_forward_CL)
		.def("test_backprop_CL", &CNN1::test_backprop_CL)
		.def("test_forward_RLU", &CNN1::test_forward_RLU)
		.def("test_backprop_RLU", &CNN1::test_backprop_RLU)
		.def("test_forward_MP", &CNN1::test_forward_MP)
		.def("test_backprop_MP", &CNN1::test_backprop_MP)
		.def("test_forward_SM", &CNN1::test_forward_SM)
		.def("test_backprop_SM", &CNN1::test_backprop_SM)
		.def("forward", &CNN1::forward_python)
		.def("backprop", &CNN1::backprop_python)
		.def("train", &CNN1::train_python)
		.def("save", &CNN1::save)
		.def("load", &CNN1::load)
		.def("forward_la", &CNN1::forward_la_python)
	;
}
