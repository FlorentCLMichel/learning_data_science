SoftMax::SoftMax(int input_len, int output_len, 
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
		
void SoftMax::save(ofstream& file){
	save_2d_array(weights, file);
	save_vector(biases, file);
	save_vector(exp_totals, file);
	save_3d_array(last_input, file);
}

void SoftMax::load(ifstream& file){
	d2_array_type weights_ = load_2d_array(file);
	auto shape_weights = weights_.shape();
	weights.resize(boost::extents[shape_weights[0]][shape_weights[1]]);
	weights = weights_;
	
	biases = load_vector<double>(file);
	
	exp_totals = load_vector<double>(file);

	d3_array_type last_input_ = load_3d_array(file);
	auto shape_last_input = last_input_.shape();
	last_input.resize(boost::extents[shape_last_input[0]][shape_last_input[1]][shape_last_input[2]]);
	last_input = last_input_;
}

vector<double> SoftMax::forward(d3_array_type &input){
	
	// clear the vector of totals 
	exp_totals.clear();

	vector<double> output;
	int n_nodes = weights.shape()[0]; // number of nodes

	// number and dimensions of images
	int nim = input.shape()[0];
	int h = input.shape()[1];
	int w = input.shape()[2];

	// cache the input
	last_input.resize(boost::extents[nim][h][w]);
	last_input = input;
	
	// compute totals
	vector<double> totals;
	for(int k=0; k<n_nodes; k++){
		totals.push_back(biases[k]);
		for(int n=0; n<nim; n++){
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					totals[k] += weights[k][n*h*w+i*w+j] * input[n][i][j];
				}
			}
		}
	}

	double sum_exp_totals = 0.;
	for(int k=0; k<n_nodes; k++){
		double exp_total_k = exp(totals[k]);
		exp_totals.push_back(exp_total_k);
		sum_exp_totals += exp_total_k;
	}

	// compute the output
	for(int k=0; k<n_nodes; k++){
		output.push_back(exp_totals[k] / sum_exp_totals);
	}
	
	// return the output
	return output;
}
		
d3_array_type SoftMax::backprop(vector<double> &d_L_d_out, double learn_rate){
	
	// number of nodes
	int n_nodes = d_L_d_out.size();

	// number and dimensions of images
	int nim = last_input.shape()[0];
	int h = last_input.shape()[1];
	int w = last_input.shape()[2];

	// We use that d_L_d_out is different from zero only for i = label
	for(int i=0; i<d_L_d_out.size(); ++i){
		if(d_L_d_out[i] == 0.){
			continue;
		}
	
		double gradient = d_L_d_out[i];
		
		// compute the exponentials of the totals and sum of the results
		double sum_exp_totals = 0.;
		for(int k=0; k<n_nodes; k++){
			sum_exp_totals += exp_totals[k];
		}
	
		// Gradient of the output p[label] against totals,
		
		vector<double> d_out_d_t;
		for(int j=0; j<n_nodes; j++){
			d_out_d_t.push_back(- exp_totals[i]*exp_totals[j] / (sum_exp_totals*sum_exp_totals));
		}
		d_out_d_t[i] += exp_totals[i] / sum_exp_totals;
		
		// Gradient of the loss against total
		
		vector<double> d_L_d_t; 
		for(int j=0; j<n_nodes; j++){
			d_L_d_t.push_back(d_out_d_t[j]*gradient);
		}
		
		/* Gradients of loss against weights, biases, and input
		   We use that: 
			* d_totals[i]_d_weights[j] = last_input for i = j and 0 otherwise
			* d_totals_d_bias is the identity matrix
			* d_totals_d_inputs = weights                        */				

		d2_array_type d_L_d_w(boost::extents[n_nodes][nim*h*w]);
		d3_array_type d_L_d_inputs(boost::extents[last_input.shape()[0]][last_input.shape()[1]][last_input.shape()[2]]);
		
		for(int n=0; n<nim; n++){
			for(int i=0; i<h; i++){
				for(int j=0; j<w; j++){
					d_L_d_inputs[n][i][j] = 0.;
					for(int l=0; l<n_nodes; l++){
						d_L_d_w[l][n*h*w+i*w+j] = d_L_d_t[l] * last_input[n][i][j];
						
						d_L_d_inputs[n][i][j] += d_L_d_t[l] * weights[l][n*h*w+i*w+j];
		
						// update weights
						weights[l][n*h*w+i*w+j] -= learn_rate * d_L_d_w[l][n*h*w+i*w+j];
					}
				}
			}
		}

		// update biases
		for(int l=0; l<n_nodes; l++){
			biases[l] -= learn_rate * d_L_d_t[l]; 
		}

		// return the gradient of the loss function against the input
		return d_L_d_inputs;
	}
}
