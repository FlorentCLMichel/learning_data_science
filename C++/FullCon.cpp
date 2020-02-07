FullCon::FullCon(int input_len, int output_len, 
	    	std::mt19937 gen, std::normal_distribution<> dis) : 
		weights(boost::extents[output_len][input_len])
{
	// initialize the biases to 0 and the weights to random values
	for(int i=0; i<output_len; i++){
		biases.push_back(0.);
		for(int j=0; j<input_len; j++){
			weights[i][j] = dis(gen) / input_len;
		}
	}
}
		
void FullCon::save(ofstream& file){
	save_2d_array(weights, file);
	save_vector(biases, file);
	save_vector(output, file);
	save_vector(last_input, file);
}

void FullCon::load(ifstream& file){
	d2_array_type weights_ = load_2d_array(file);
	auto shape_weights = weights_.shape();
	weights.resize(boost::extents[shape_weights[0]][shape_weights[1]]);
	weights = weights_;
	
	biases = load_vector<double>(file);
	
	output = load_vector<double>(file);

	vector<double> last_input = load_vector<double>(file);
}

// sigmoid function
double sigmoid(double x){
	return (1./(1.+exp(-x)));
}

vector<double> FullCon::forward(vector<double> &input){
	
	// clear the vector of outputs
	output.clear();

	vector<double> totals;
	int n_nodes = weights.shape()[0]; // number of nodes

	// cache the input
	last_input = input;
	
	// compute totals
	for(int k=0; k<n_nodes; k++){
		totals.push_back(biases[k]);
		for(int n=0; n<input.size(); n++){
			totals[k] += weights[k][n] * input[n];
		}
		output.push_back(sigmoid(totals[k]));
	}
	
	// return the output
	return output;
}
		
vector<double> FullCon::backprop(vector<double> &d_L_d_out, double learn_rate){
	
	// number of nodes and input size
	int n_nodes = d_L_d_out.size();
	int len_input = last_input.size();

	/* Gradients of loss against weights, biases, and input
	   We use that: 
		* d_totals[i]_d_weights[j] = last_input for i = j and 0 otherwise
		* d_totals_d_bias is the identity matrix
		* d_totals_d_inputs = weights                        */				

	d2_array_type d_L_d_w(boost::extents[n_nodes][last_input.size()]);
	vector<double> d_L_d_inputs;
	
	for(int n=0; n<last_input.size(); n++){
		d_L_d_inputs.push_back(0.);
		for(int l=0; l<n_nodes; l++){
			// We use that the derivative of the sigmoid function s is s*(1-s)
			d_L_d_w[l][n] = d_L_d_out[l] * output[l] * (1.-output[l]) * last_input[n];
			
			d_L_d_inputs[n] += d_L_d_out[l] * output[l] * (1.-output[l]) * weights[l][n];
	
			// update weights
			weights[l][n] -= learn_rate * d_L_d_w[l][n];
		}
	}

	// update biases
	for(int l=0; l<n_nodes; l++){
		biases[l] -= learn_rate * d_L_d_out[l] * output[l] * (1.-output[l]); 
	}

	// return the gradient of the loss function against the input
	return d_L_d_inputs;
}
