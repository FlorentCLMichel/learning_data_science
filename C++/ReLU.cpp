void ReLU::save(ofstream& file){
	save_3d_array(last_input, file);
}

void ReLU::load(ifstream& file){
	d3_array_type last_input_ = load_3d_array(file);
	auto shape_last_input = last_input_.shape();
	last_input.resize(boost::extents[shape_last_input[0]][shape_last_input[1]][shape_last_input[2]]);
	last_input = last_input_;
}
		
d3_array_type ReLU::forward(d3_array_type &input){

	int nim = input.shape()[0]; // number of images
	int h = input.shape()[1]; // height
	int w = input.shape()[2]; // width
	
	// cache the input
	last_input.resize(boost::extents[nim][h][w]);
	last_input = input;
	
	// initialize and compute the output
	d3_array_type output(boost::extents[nim][h][w]);
	for(int n=0; n<nim; n++){
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				double el = input[n][i][j];
				if(el >= 0.){
					output[n][i][j] = input[n][i][j];
				} else {
					output[n][i][j] = 0.;
				}
			}
		}
	}
	return output;
}
		
d3_array_type ReLU::backprop(d3_array_type &d_L_d_out){
	
	// number and dimensions of the images in the last input
	int nim = last_input.shape()[0]; 
	int h = last_input.shape()[1];
	int w = last_input.shape()[2];

	// gradient of the loss function against the input
	d3_array_type d_L_d_input(boost::extents[nim][h][w]);
	for(int n=0; n<nim; n++){
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				double el = last_input[n][i][j];
				if(el >= 0.){
					d_L_d_input[n][i][j] = d_L_d_out[n][i][j];
				} else {
					d_L_d_input[n][i][j] = 0.;
				}
			}
		}
	}
	return d_L_d_input;
}
