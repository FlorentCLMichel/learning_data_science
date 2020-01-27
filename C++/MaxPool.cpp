void MaxPool::save(ofstream& file){
	file << size << sep_line;
	save_3d_array(last_input_maxima, file);
}

void MaxPool::load(ifstream& file){
	file >> size >> sep_line;
	d3_array_type last_input_ = load_3d_array(file);
	auto shape_last_input = last_input_.shape();
	last_input_maxima.resize(boost::extents[shape_last_input[0]][shape_last_input[1]][shape_last_input[2]]);
	last_input_maxima = last_input_;
}

d3_array_type MaxPool::forward(d3_array_type &input){

	// number and dimensions of images
	int nim = input.shape()[0];
	int h = input.shape()[1];
	int w = input.shape()[2];
	
	// dimensions of the output images
	int h_out = (int) h / size;
	int w_out = (int) w / size;

	// where the maxima of the input are located
	last_input_maxima.resize(boost::extents[nim][h][w]);
	fill(last_input_maxima.data(), last_input_maxima.data() + last_input_maxima.num_elements(), 0.);

	// compute the output
	d3_array_type output(boost::extents[nim][h_out][w_out]);
	for(int n=0; n<nim; n++){
		for(int i=0; i<h_out; i++){
			for(int j=0; j<w_out; j++){
				double the_max = input[n][i*size][j*size]; // will contain the maximum for this pool
				int ii_max = 0;
				int jj_max = 0;
				for(int ii=0; ii<size; ii++){
					for(int jj=0; jj<size; jj++){
						double el = input[n][i*size+ii][j*size+jj];
						if(el > the_max){
							the_max = el;
							ii_max = ii;
							jj_max = jj;
						}
					}
				}
				// retain only the maximum value in the current pool
				output[n][i][j] = the_max; 
				last_input_maxima[n][i*size+ii_max][j*size+jj_max] = 1;
			}
		}
	}

	// return the output
	return output;
}

d3_array_type MaxPool::backprop(d3_array_type &d_L_d_out){

	// number and dimensions of images in the last input
	int nim = last_input_maxima.shape()[0];
	int h = last_input_maxima.shape()[1];
	int w = last_input_maxima.shape()[2];
	
	// initialize the gradient of the loss function against the input
	d3_array_type d_L_d_input(boost::extents[nim][h][w]);
	fill(d_L_d_input.data(), d_L_d_input.data() + d_L_d_input.num_elements(), 0.);

	// set the non-zero values where the maxima of the input where located
	for(int n=0; n < (int) nim; n++){
		for(int i=0; i < (int) h / size; i++){
			for(int j=0; j < (int) w / size; j++){
				for(int ii=0; ii<size; ii++){
					for(int jj=0; jj<size; jj++){
						d_L_d_input[n][i*size+ii][j*size+jj] = d_L_d_out[n][i][j] * last_input_maxima[n][i*size+ii][j*size+jj];
					}
				}
			}
		}
	}
	
	// return the gradient against the input
	return d_L_d_input;
}
