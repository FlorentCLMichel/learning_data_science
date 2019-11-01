#include <stdlib.h>
#include <math.h>
#include <time.h>

// random integer between 0 and Nmax
int rand_int(int Nmax){
	int divisor = RAND_MAX/(Nmax+1);
	int retval;
	do{
		retval = rand() / divisor;
	} while (retval > Nmax);
	return retval;
}

// mean value of the entries in the vector vec
// vec: pointer to double
// length: int 
double mean(double *vec, int length){
	double sum = 0;
	for(int i=0; i<length; i++){
		sum = sum + vec[i];
	}
	return sum / length;
}

// product of x and y
// x: pointer to double
// y: pointer to double
// length: int 
double* multiply_vec(double *x, double *y, int length){
	double *res = malloc(length*sizeof(double));
	for(int i=0; i<length; i++){
		res[i] = x[i]*y[i];
	}
	return res;
}

double std(double *x, int length)
{
	double mean_x = mean(x, length);
	double *xx = multiply_vec(x, x, length);
	double mean_xx = mean(xx, length);
	free(xx);
	return sqrt(mean_xx - mean_x*mean_x);
}

// correlation coefficient of x and y
// x: pointer to double
// y: pointer to double
// length: int 
double corr(double *x, double *y, int length){
	double mean_x = mean(x, length);
	double mean_y = mean(y, length);
	double *xx = multiply_vec(x, x, length);
	double *yy = multiply_vec(y, y, length);
	double *xy = multiply_vec(x, y, length);
	double mean_xx = mean(xx, length);
	double mean_yy = mean(yy, length);
	double mean_xy = mean(xy, length);
	double var_x = mean_xx - mean_x*mean_x;
	double var_y = mean_yy - mean_y*mean_y;
	double cov_xy = mean_xy - mean_x*mean_y;
	free(xx);
	free(yy);
	free(xy);
	return cov_xy/sqrt(var_x*var_y);
}

// correlation coefficient of a random reshuffling of x and y with replacements
double corr_rand(double *x, double *y, int length){
	double *x_r = malloc(length*sizeof(double));
	double *y_r = malloc(length*sizeof(double));
	int index;
	for(int i=0; i<length; i++){
		index = rand_int(length-1);
		x_r[i] = x[index];
		y_r[i] = y[index];
	}
	double res = corr(x_r, y_r, length);
	free(x_r);
	free(y_r);
	return res;
}

// standard deviation of the correlation coefficient
// n is the number of reshufflings
double std_corr(double *x, double *y, int length, int n){
	if(n==0){
		n = length;
	}
	srand(time(NULL));
	double *corrs = malloc(n*sizeof(double));
	for(int i=0; i<n; i++){
		corrs[i] = corr_rand(x, y, length);
	}
	double res = std(corrs, n);
	free(corrs);
	return res;
}
