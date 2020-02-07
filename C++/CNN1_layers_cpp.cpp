// separating characters for the save and load functions
char sep_val = '*';
char sep_line = '$';

// define types for 2d and 3d arrays
typedef boost::multi_array<double, 2> d2_array_type;
typedef boost::multi_array<double, 3> d3_array_type;

// converts a 3d numpy array to a boost multiarray
d3_array_type d3_numpy_to_multi_array(np::ndarray a){
	int h = a.shape(0);
	int w = a.shape(1);
	int p = a.shape(2);
	d3_array_type A(boost::extents[h][w][p]);
	for(d3_array_type::index i = 0; i<h; i++){
		for(d3_array_type::index j = 0; j<w; j++){
			for(d3_array_type::index k = 0; k<p; k++){
				A[i][j][k] = p::extract<double>(a[i][j][k]);
			}
		}
	}
	return A;
}

// converts a 3d array to a numpy array
np::ndarray d3array_to_numpy(d3_array_type A){
	const long unsigned int* shape_A = A.shape();
	p::tuple shape = p::make_tuple(shape_A[0], shape_A[1], shape_A[2]);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray a = np::zeros(shape, dtype);
	for(int i=0; i<A.shape()[0]; i++){
		for(int j=0; j<A.shape()[1]; j++){
			for(int k=0; k<A.shape()[2]; k++){
				a[i][j][k] = A[i][j][k];
			}
		}
	}
	return a;
}

// converts a 2d numpy array to a boost multiarray
d2_array_type d2_numpy_to_multi_array(np::ndarray a){
	int h = a.shape(0);
	int w = a.shape(1);
	d2_array_type A(boost::extents[h][w]);
	for(d3_array_type::index i = 0; i<h; i++){
		for(d3_array_type::index j = 0; j<w; j++){
			A[i][j] = p::extract<double>(a[i][j]);
		}
	}
	return A;
}

// converts a 2d array to a numpy array
np::ndarray d2array_to_numpy(d2_array_type A){
	const long unsigned int* shape_A = A.shape();
	p::tuple shape = p::make_tuple(shape_A[0], shape_A[1]);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray a = np::zeros(shape, dtype);
	for(int i=0; i<A.shape()[0]; i++){
		for(int j=0; j<A.shape()[1]; j++){
			a[i][j] = A[i][j];
		}
	}
	return a;
}

// converts a vector to a numpy array
np::ndarray vector_to_numpy(vector<double> vec){
	p::tuple shape = p::make_tuple(vec.size());
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray a = np::zeros(shape, dtype);
	for(int i=0; i<vec.size(); i++){
		a[i] = vec[i];
	}
	return a;
}

// converts a numpy array to a vector
vector<double> numpy_to_vector(np::ndarray a){
	int h = a.shape(0);
	vector<double> A;
	for(int i=0; i<h; i++){
		A.push_back(p::extract<double>(a[i]));
	}
	return A;
}

// converts a python list of int to a vector
vector<int> int_list_to_vector(p::list& ls){
	vector<int> vec;
	for(int i=0; i<len(ls); i++){
		vec.push_back(p::extract<int>(ls[i]));
	}
	return vec;
}

// write a vector to a file, with each entry preceded by '*', and print a '$' at the end
template <class T>
void save_vector(vector<T> &vec, ofstream& file){
	for(int i=0; i<vec.size(); i++){
		file << sep_val << vec[i];
	}
	file << sep_line;
}

// read a vector from a file, stopping at the first '$'
template <typename T>
vector<T> load_vector(ifstream& file){
	vector<T> res;
	char c;
	T x;
	file >> c;
	while(c!='$'){
		file >> x >> c;
		res.push_back(x);
	}
	return res;
}

// write a 2d array to a file (shape followed by data), with entries preceded 
// by '*' except the first value of the shape, and print a '$' at the end of 
// the shape and data
void save_2d_array(d2_array_type &arr, ofstream& file){
	auto shape = arr.shape();
	file << shape[0] << sep_val << shape[1] << sep_line;
	for(int i=0; i<shape[0]; i++){
		for(int j=0; j<shape[1]; j++){
			file << sep_val << arr[i][j];
		}
	}
	file << sep_line;
}

// load a 2d array from a file
d2_array_type load_2d_array(ifstream& file){
	int h;
	int w;
	char c;
	file >> h >> c >> w >> c;
	d2_array_type res(boost::extents[h][w]);
	double x;
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			file >> c >> res[i][j];
		}
	}
	file >> c;
	return res;
}

// write a 3d array to a file (shape followed by data), with entries preceded 
// by '*' except the first value of the shape, and print a '$' at the end of 
// the shape and data
void save_3d_array(d3_array_type &arr, ofstream& file){
	auto shape = arr.shape();
	file << shape[0] << sep_val << shape[1]  << sep_val << shape[2] << sep_line;
	for(int i=0; i<shape[0]; i++){
		for(int j=0; j<shape[1]; j++){
			for(int k=0; k<shape[2]; k++){
				file << sep_val << arr[i][j][k];
			}
		}
	}
	file << sep_line;
}

// load a 3d array from a file
d3_array_type load_3d_array(ifstream& file){
	int h;
	int w;
	int d;
	char c;
	file >> h >> c >> w >> c >> d >> c;
	d3_array_type res(boost::extents[h][w][d]);
	double x;
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			for(int k=0; k<d; k++){
				file >> c >> res[i][j][k];
			}
		}
	}
	file >> c;
	return res;
}

/*
a convolution layer class
all filters are assumed to be square and have the same size

size_filters: size of each filter
num_filters: number of filters
*/
class ConvLayer{

	private:
		
		int size_filters;
		int num_filters;
		d3_array_type filters;
		d3_array_type last_input;
		
		// A useful loop for the forward propagation
		static void loop_1(int size_filters, int num_filters, int f, int nim, int h_out, int w_out, double* output, double* input, double* filters);
		
		// A useful loop for the backpropagation
		static void loop_2(int f, int nim, int h, int w, int h_out, int w_out, int size_filters, int num_filters, double* last_input, double* filters, double* d_L_d_input, double* d_L_d_filters, double* d_L_d_out);

	public: 

		ConvLayer(){}

		ConvLayer(int size_filters_, int num_filters_, 
			std::mt19937 gen, std::normal_distribution<> dis);
		
		// save the layer to a file
		void save(ofstream& file);

		// load the layer from a file
		void load(ifstream& file);

		// Performs a forward pass of the convolution layer on the input
		d3_array_type forward(d3_array_type &input);

		// backward pass of the convolution layer
		// d_L_d_out: gradient of the loss function against the output
		d3_array_type backprop(d3_array_type &d_L_d_out, double learn_rate);
		
		// test function: forward
		np::ndarray test_forward(np::ndarray input){
			d3_array_type array = d3_numpy_to_multi_array(input);
			d3_array_type array2 = forward(array);
			return d3array_to_numpy(array2);
		}
		
		// test function: backprop
		np::ndarray test_backprop(np::ndarray input, double learn_rate){
			d3_array_type array = d3_numpy_to_multi_array(input);
			d3_array_type array2 = backprop(array, learn_rate);
			return d3array_to_numpy(array2);
		}
};

#include "CNN1_ConvLayer.cpp"

// A ReLU layer
class ReLU{
	
	public: 

		d3_array_type last_input;
		
		// save the layer to a file
		void save(ofstream& file);

		// load the layer from a file
		void load(ifstream& file);
		
		// forward propagation
		d3_array_type forward(d3_array_type &input);

		// backpropagation
		// d_L_d_out: gradient of the loss function against the output
		d3_array_type backprop(d3_array_type &d_L_d_out);

		// test function: forward
		np::ndarray test_forward(np::ndarray input){
			d3_array_type array = d3_numpy_to_multi_array(input);
			d3_array_type array2 = forward(array);
			return d3array_to_numpy(array2);
		}
		
		// test function: backprop
		np::ndarray test_backprop(np::ndarray input){
			d3_array_type array = d3_numpy_to_multi_array(input);
			d3_array_type array2 = backprop(array);
			return d3array_to_numpy(array2);
		}
};

#include "ReLU.cpp"

// A max pooling layer with square pools
// size: size of the pools 
class MaxPool{
	
	private: 

		int size;
		d3_array_type last_input_maxima;
	
	public: 

		MaxPool(){}
		
		MaxPool(int size_){
			size = size_;
		}

		// save the layer to a file
		void save(ofstream& file);

		// load the layer from a file
		void load(ifstream& file);
		
		// forward pass
		d3_array_type forward(d3_array_type &input);
		
		// backpropagation
		// d_L_d_out: gradient of the loss function against the output
		d3_array_type backprop(d3_array_type &d_L_d_out);
		
		// test function: forward
		np::ndarray test_forward(np::ndarray input){
			d3_array_type array = d3_numpy_to_multi_array(input);
			d3_array_type array2 = forward(array);
			return d3array_to_numpy(array2);
		}
		
		// test function: backprop
		np::ndarray test_backprop(np::ndarray input){
			d3_array_type array = d3_numpy_to_multi_array(input);
			d3_array_type array2 = backprop(array);
			return d3array_to_numpy(array2);
		}
};

#include "MaxPool.cpp"

// A standard fully-connected layer with sigmoid activation function
class FullCon{
	
	private: 

		d2_array_type weights;
		vector<double> biases;
		vector<double> output;
		
	public:	

		FullCon(){}

		FullCon(int input_len, int output_len, 
			    	std::mt19937 gen, std::normal_distribution<> dis);

		vector<double> last_input;

		// save the layer to a file
		void save(ofstream& file);

		// load the layer from a file
		void load(ifstream& file);
	
		// forward pass
		vector<double> forward(vector<double> &input);
			
		// Performs a backward pass of the softmax layer and returns the loss 
		// gradient with respect to the input.
		// d_L_d_out: gradient of the loss function against the output
		vector<double> backprop(vector<double> &d_L_d_out, double learn_rate);

		// test function: forward
		np::ndarray test_forward(np::ndarray input){
			vector<double> array = numpy_to_vector(input);
			vector<double> array2 = forward(array);
			return vector_to_numpy(array2);
		}

		// test function: backprop
		np::ndarray test_backprop(np::ndarray d_L_d_out, double learn_rate){
			vector<double> d_L_d_out_ = numpy_to_vector(d_L_d_out);
			vector<double> d_L_d_input = backprop(d_L_d_out_, learn_rate);
			return vector_to_numpy(d_L_d_input);
		}

};

#include "FullCon.cpp"

// A standard fully-connected layer with softmax activation
class SoftMax{
	
	private: 

		d2_array_type weights;
		vector<double> biases;
		vector<double> exp_totals;
		
	public:	

		SoftMax(){}

		SoftMax(int input_len, int output_len, 
			    	std::mt19937 gen, std::normal_distribution<> dis);

		vector<double> last_input;

		// save the layer to a file
		void save(ofstream& file);

		// load the layer from a file
		void load(ifstream& file);
	
		// forward pass
		vector<double> forward(vector<double> &input);
			
		// Performs a backward pass of the softmax layer and returns the loss 
		// gradient with respect to the input.
		// d_L_d_out: gradient of the loss function against the output
		vector<double> backprop(vector<double> &d_L_d_out, double learn_rate);

		// test function: forward
		np::ndarray test_forward(np::ndarray input){
			vector<double> array = numpy_to_vector(input);
			vector<double> array2 = forward(array);
			return vector_to_numpy(array2);
		}

		// test function: backprop
		np::ndarray test_backprop(np::ndarray d_L_d_out, double learn_rate){
			vector<double> d_L_d_out_ = numpy_to_vector(d_L_d_out);
			vector<double> d_L_d_input = backprop(d_L_d_out_, learn_rate);
			return vector_to_numpy(d_L_d_input);
		}

};

#include "SoftMax.cpp"
