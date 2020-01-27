ConvLayer::ConvLayer(int size_filters_, int num_filters_, 
	std::mt19937 gen, std::normal_distribution<> dis) :
		filters(boost::extents[num_filters_][size_filters_][size_filters_])
	{
	size_filters = size_filters_;
	num_filters = num_filters_;

	long size2 = size_filters*size_filters;

	for(int i=0; i<num_filters; i++){
		for(int j=0; j<size_filters; j++){
			for(int k=0; k<size_filters; k++){
				filters[i][j][k] = dis(gen)/size2;
			}
		}
	}
}

void ConvLayer::save(ofstream& file){
	file << size_filters << sep_val << num_filters << sep_line;
	save_3d_array(filters, file);
	save_3d_array(last_input, file);
}
		
void ConvLayer::load(ifstream& file){
	file >> size_filters >> sep_val >> num_filters >> sep_line;
	d3_array_type filters_ = load_3d_array(file);
	auto shape_filters = filters_.shape();
	filters.resize(boost::extents[shape_filters[0]][shape_filters[1]][shape_filters[2]]);
	filters = filters_;
	d3_array_type last_input_ = load_3d_array(file);
	auto shape_last_input = last_input_.shape();
	last_input.resize(boost::extents[shape_last_input[0]][shape_last_input[1]][shape_last_input[2]]);
	last_input = last_input_;
}
		
void ConvLayer::loop_1(int size_filters, int num_filters, int f, int nim, int h_out, int w_out, double* output, double* input, double* filters){
	for(int i = 0; i < h_out; i++){
		for(int j = 0; j < w_out; j++){
			for(int n = 0; n < nim; n++){
				for(int ii = 0; ii < size_filters; ii++){
					for(int jj = 0; jj < size_filters; jj++){
						output[(n * num_filters + f)*h_out*w_out + i*w_out + j] += input[n*(h_out+size_filters-1)*(w_out+size_filters-1) + (i+ii)*(w_out+size_filters-1) + (j+jj)] * filters[f*size_filters*size_filters + ii*size_filters + jj];
					}
				}
			}
		}
	}
}
		
d3_array_type ConvLayer::forward(d3_array_type &input)
{
	int nim = input.shape()[0]; // number of images

	// height and width of the input images
	int h = input.shape()[1];
	int w = input.shape()[2];
	
	// height and width of the output images
	int h_out = h-size_filters+1; 
	int w_out = w-size_filters+1; 
	
	// initialize the output
	d3_array_type output(boost::extents[nim * num_filters][h_out][w_out]);
	fill(output.data(), output.data() + output.num_elements(), 0.);
	
	// cache the input
	last_input.resize(boost::extents[nim][h][w]);
	last_input = input;

	// compute the output
	thread thread_[num_filters];
	for(int f = 0; f < num_filters; f++){
		// for(int i = 0; i < h_out; i++){
		// 	for(int j = 0; j < w_out; j++){
		// 		for(int n = 0; n < nim; n++){
		// 			for(int ii = 0; ii < size_filters; ii++){
		// 				for(int jj = 0; jj < size_filters; jj++){
		// 					output[n * num_filters + f][i][j] += input[n][i+ii][j+jj] * filters[f][ii][jj];
		// 				}
		// 			}
		// 		}
		// 	}
		// }
		thread_[f] = thread(loop_1, size_filters, num_filters, f, nim, h_out, w_out, output.data(), input.data(), filters.data());
	}
	for(int f=0; f<num_filters; f++){
		thread_[f].join();
	}

	// return the output
	return output;
}

void ConvLayer::loop_2(int f, int nim, int h, int w, int h_out, int w_out, int size_filters, int num_filters, double* last_input, double* filters, double* d_L_d_input, double* d_L_d_filters, double* d_L_d_out){
	for(int ii=0; ii<size_filters; ii++){
		for(int jj=0; jj<size_filters; jj++){
			for(int n=0; n<nim; n++){
				for(int i=0; i<h_out; i++){
					for(int j=0; j<w_out; j++){
						d_L_d_filters[f*size_filters*size_filters + ii*size_filters + jj] += d_L_d_out[(n*num_filters+f)*h_out*w_out + i*w_out + j] * last_input[n*h*w + (i+ii)*w + j+jj];
						d_L_d_input[n*h*w + (i+ii)*w + j+jj] += d_L_d_out[(n*num_filters+f)*h_out*w_out + i*w_out + j] * filters[f*size_filters*size_filters + ii*size_filters + jj];
					}
				}
			}
		}
	}
}
		
d3_array_type ConvLayer::backprop(d3_array_type &d_L_d_out, double learn_rate){

	// get the number of imagines, height, and wifth of the last input
	int nim = last_input.shape()[0];
	int h = last_input.shape()[1];
	int w = last_input.shape()[2];
	
	// height and width of the output images
	int h_out = h-size_filters+1; 
	int w_out = w-size_filters+1; 

	// initialize the gradients against the filters and input
	d3_array_type d_L_d_filters(boost::extents[num_filters][size_filters][size_filters]);
	fill(d_L_d_filters.data(), d_L_d_filters.data() + d_L_d_filters.num_elements(), 0.);
	d3_array_type d_L_d_input(boost::extents[nim][h][w]);
	fill(d_L_d_input.data(), d_L_d_input.data() + d_L_d_input.num_elements(), 0.);

	thread thread_[num_filters];
	for(int f=0; f<num_filters; f++){
		// for(int ii=0; ii<size_filters; ii++){
		// 	for(int jj=0; jj<size_filters; jj++){
		// 		for(int n=0; n<nim; n++){
		// 			for(int i=0; i<h_out; i++){
		// 				for(int j=0; j<w_out; j++){
		// 					d_L_d_filters[f][ii][jj] += d_L_d_out[n*num_filters+f][i][j] * last_input[n][i+ii][j+jj];
		// 					d_L_d_input[n][i+ii][j+jj] += d_L_d_out[n*num_filters+f][i][j] * filters[f][ii][jj];
		// 				}
		// 			}
		// 		}
		// 	}
		// }
		thread_[f] = thread(loop_2, f, nim, h, w, h_out, w_out, size_filters, num_filters, last_input.data(), filters.data(), d_L_d_input.data(), d_L_d_filters.data(), d_L_d_out.data());
	}
	for(int f=0; f<num_filters; f++){
		thread_[f].join();
	}

	// update the filters
	for(int f=0; f<num_filters; f++){
		for(int ii=0; ii<size_filters; ii++){
			for(int jj=0; jj<size_filters; jj++){
				filters[f][ii][jj] -= learn_rate * d_L_d_filters[f][ii][jj];
			}
		}
	}
	
	// return the gradient of the loss function against the input
	return d_L_d_input;
}
