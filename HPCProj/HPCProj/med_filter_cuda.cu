#include "stdafx.h"
#include "med_filter.h"
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

__global__ void mf_cuda_sm(unsigned int *padded_array, int padded_array_size, int window_size, unsigned int* filtered_array, int original_size, int mid){
	
	__shared__ int smem[BLOCK_WIDTH*BLOCK_HEIGHT];
	
	unsigned int window[1000];
	
	//int x = blockDim.x * BLOCK_WIDTH - 2 * mid + threadIdx.x;
	//int y = blockDim.y * BLOCK_WIDTH - 2 * mid + threadIdx.y;
	int x = blockIdx.x * (BLOCK_WIDTH - (2 * mid)) + threadIdx.x;
	int y = blockIdx.y * (BLOCK_WIDTH - (2 * mid)) + threadIdx.y;

	int x_pad = blockIdx.x * BLOCK_WIDTH + threadIdx.x - blockIdx.x*mid*2 ;
	int y_pad = blockIdx.y * BLOCK_WIDTH + threadIdx.y - blockIdx.y*mid*2 ;

	if (x_pad >= padded_array_size || y_pad >= padded_array_size)
		return;

	unsigned int index = y_pad * padded_array_size + x_pad;
	unsigned int bindex = threadIdx.y * blockDim.x + threadIdx.x;

	/*if (x == original_size + mid && y == original_size + mid){
		printf("[%d], [%d] ", x_pad, y_pad);
	}*/

	smem[bindex] = padded_array[index];
	//printf("[%d], [%d] ",y, x);
	__syncthreads();

	if ((threadIdx.x >= mid) && (threadIdx.x < (BLOCK_WIDTH - mid)) &&
		(threadIdx.y >= mid) && (threadIdx.y < (BLOCK_HEIGHT - mid)) &&
		((x) < original_size + mid) && ((y) < original_size + mid)){
		/*printf("[%d][%d] : %d \n\t\t%d, %d, %d\n\t\t%d, %d, %d\n\t\t%d, %d, %d\n", y, x, padded_array[index],
			smem[bindex - (BLOCK_WIDTH)-1], smem[bindex - BLOCK_WIDTH], smem[bindex - BLOCK_WIDTH + 1],
			smem[bindex - 1], smem[bindex], smem[bindex + 1],
			smem[bindex + BLOCK_WIDTH - 1], smem[bindex + BLOCK_WIDTH], smem[bindex + BLOCK_WIDTH + 1]
			);*/
		//printf("x: %d | y: %d, ", threadIdx.x, threadIdx.y);
		int sum = 0;
		for (int dy = -mid; dy <= mid; dy++){
			for (int dx = -mid; dx <= mid; dx++){
				window[sum] = smem[bindex + (dy*blockDim.x) + dx];
				//printf("%d, ", window[sum]);
				sum++;
				
			}
		}

		for (int i = 0; i < (window_size*window_size + 1) / 2; ++i) {
			int minval = i;
			for (int l = i + 1; l < (window_size*window_size); ++l)
				if (window[l] < window[minval])
					minval = l;

			unsigned int temp = window[i];
			window[i] = window[minval];
			window[minval] = temp;
		}

		//Results
		//filtered_array[((blockIdx.y * blockDim.y) + threadIdx.y - mid) * original_size + ((blockIdx.x * blockDim.x) + threadIdx.x - mid)] = 1; //window[(window_size*window_size) / 2];
		filtered_array[(y - mid) * original_size + (x - mid)] = window[(window_size*window_size) / 2]; // smem[bindex];
	}
};

/*__global__ void mf_cuda_sm(unsigned int *padded_array, int padded_array_size, int window_size, unsigned int* filtered_array, int original_size, int mid) {

	//__shared__ unsigned int window[BLOCK_HEIGHT*BLOCK_WIDTH][9];
	
	int count;

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int tid = threadIdx.y * blockDim.x + threadIdx.x;

	if ((x >= padded_array_size - mid) || (y >= padded_array_size - mid) || (x < mid) || (y < mid))
		return;

	count = 0;
	for (int c = y - mid; c <= y + mid; c++) {
		for (int r = x - mid; r <= x + mid; r++) {
			window[tid][count] = padded_array[padded_array_size*c + r];
			//printf("%d, ", window[tid][count]);
			count++;
		}
	};
	for (int i = 0; i < (window_size*window_size + 1) / 2; ++i) {
		int minval = i;
		for (int l = i + 1; l < (window_size*window_size); ++l)
			if (window[l] < window[minval])
				minval = l;

		unsigned int temp = window[tid][i];
		window[tid][i] = window[tid][minval];
		window[tid][minval] = temp;
	};
	__syncthreads();
	filtered_array[original_size*(y - mid) + (x - mid)] = window[tid][(window_size*window_size) / 2];
}*/

__global__ void mf_cuda(unsigned int *padded_array, int padded_array_size, int window_size, unsigned int* filtered_array, int original_size, int mid)
{

	unsigned int window[441];
	int count;

	int x = blockDim.x * blockIdx.x + threadIdx.x + mid;
	int y = blockDim.y * blockIdx.y + threadIdx.y + mid;

	if ((x >= original_size + mid) && (y >= original_size + mid))
		return;

	count = 0;
	for (int c = y - mid; c <= y + mid; c++) {
		for (int r = x - mid; r <= x + mid; r++) {
			window[count] = padded_array[padded_array_size*c + r];
			count++;
		}
	}

	for (int i = 0; i < (window_size*window_size + 1) / 2; ++i) {
		int minval = i;
		for (int l = i + 1; l < (window_size*window_size); ++l)
			if (window[l] < window[minval])
				minval = l;

		unsigned int temp = window[i];
		window[i] = window[minval];
		window[minval] = temp;
	}

	//Results
	filtered_array[original_size*(y - mid) + (x - mid)] = window[(window_size*window_size) / 2];
};

unsigned int *array_padder(int original_size, int window_size, unsigned int *original_array){
	
	int pad_val = (window_size - 1) / 2;
	int padded_size = (original_size + window_size - 1);

	unsigned int* padded_array;
	padded_array = new unsigned int[padded_size*padded_size]();

	for (int i = 0; i < padded_size*padded_size; i++){
		padded_array[i] = 0;
	}
	
	for (int i = pad_val; i < padded_size - pad_val; ++i){
		for (int j = pad_val; j < padded_size - pad_val; ++j){
			padded_array[padded_size*i+j] = original_array[original_size*(i - pad_val)+(j - pad_val)];
		}
	}

	//Symmetry
	for (int x = pad_val; x < pad_val + original_size; ++x){
		int count = 0;
		for (int top = pad_val - 1; top >= 0; --top){
			padded_array[padded_size*top+x] = original_array[original_size*count+(x - pad_val)];
			padded_array[padded_size*x + top] = original_array[original_size*(x - pad_val)+count];
			count++;
		}
	}

	for (int y = original_size - 1; y >= 0; --y){
		for (int space = 0; space < pad_val; space++){
			padded_array[padded_size*(y + pad_val)+(padded_size - pad_val + space)] = original_array[original_size*y+(original_size - 1 - space)];
		}
	}

	int tmpVal = original_size - 1;
	for (int y = original_size + pad_val; y < padded_size; ++y){
		for (int x = 0 + pad_val; x < original_size + pad_val; ++x){
			padded_array[padded_size*y+x] = original_array[original_size*tmpVal+(x - pad_val)];
		}
		tmpVal--;
	}
	//EndSymmetry

	/*for (int i = 0; i < padded_size; i++){
		for (int j = 0; j < padded_size; j++){
			if (padded_array[padded_size*i + j] == 0){
				cout << padded_array[padded_size*i+j] << "\t";
			}
			else
				cout << padded_array[padded_size*i+j] << "\t";
			}
		cout << endl;
	}*/

	return padded_array;
}

void medianFilter(vector<float> bins, int window_size, unsigned int* unfiltered_array, int unfiltered_x, int unfiltered_y){
	int p, mid = (window_size - 1) / 2;

	cout << "::Create window array" << endl;

	unsigned int* window;
	window = new unsigned int[window_size*window_size]();
	for (int i = 0; i < window_size*window_size; i++){
		window[i] = 0;
	}

	cout << "::Create 2D filtered array" << endl;
	unsigned int* filtered_points;
	filtered_points = new unsigned int[unfiltered_x*unfiltered_x]();
	for (int i = 0; i < unfiltered_x*unfiltered_y; i++){
			filtered_points[i] = 0;
	}

	double startSerialPad = omp_get_wtime();
	unsigned int* padded_array = array_padder(unfiltered_x, window_size, unfiltered_array);
	double endSerialPad = omp_get_wtime();
	cout << "::Array padded:\t\t" << endSerialPad - startSerialPad << endl;

	cout << "\n==Enter serial loop==" << endl;
	double startSerial = omp_get_wtime();

	int padded_array_size = unfiltered_x + window_size - 1;

	for (int column = mid; column < padded_array_size - mid; column++)
	{
		for (int row = mid; row < padded_array_size - mid; row++)
		{
			p = 0;
			for (int c = column - mid; c <= column + mid; c++)
				for (int r = row - mid; r <= row + mid; r++)
				{
					window[p] = padded_array[padded_array_size*c+r];
					p++;
				}
			for (int i = 0; i<(window_size*window_size+1)/2; ++i) {

				int min = i;
				for (int l = i + 1; l<(window_size*window_size); ++l) 
					if (window[l] < window[min])
						min = l;

				unsigned int temp = window[i];
				window[i] = window[min];
				window[min] = temp;
			}
			filtered_points[unfiltered_x*(column - mid)+(row - mid)] = window[(window_size*window_size) / 2];

		}
	} 

	double endSerial = omp_get_wtime();

	cout << "::Loop time serial:\t\t" << endSerial - startSerial << endl;

	/*for (int i = 0; i < unfiltered_x; i++){
		for (int j = 0; j < unfiltered_y; j++){
			cout << filtered_points[unfiltered_x*i + j] << " ";
		}
		cout << endl;
	}*/

	printToFile(bins, filtered_points, unfiltered_x, unfiltered_y, "filtered_serial.csv");

	delete[] filtered_points;
	delete[] padded_array;
	delete[] window;
};

int medianFilter_CUDA(vector<float> bins, int window_size, unsigned int* unfiltered_array, int unfiltered_x, int unfiltered_y){

	/*unsigned int* window;
	window = new unsigned int[window_size*window_size]();
	for (int i = 0; i < (window_size*window_size); i++){
		window[i] = 0;
	}
	unsigned int* d_window;*/

	//unsigned int *d_unfiltered;
	int mid = (window_size - 1) / 2;

	int TILE_W = BLOCK_WIDTH - 2 * mid;
	int TILE_H = BLOCK_HEIGHT - 2 * mid;

	int padded_arr_size = unfiltered_x + window_size - 1;
	cout << "::Create 2D filtered array" << endl;

	unsigned int *result_array;
	result_array = new unsigned int[unfiltered_x*unfiltered_y]();

	unsigned int *d_filtered_array;

	double startSerialPad = omp_get_wtime();
	
	unsigned int* padded_array = array_padder(unfiltered_x, window_size, unfiltered_array);
	unsigned int *d_padded_array;
	
	double endSerialPad = omp_get_wtime();
	cout << "::Array padded:\t\t" << endSerialPad - startSerialPad << endl;

	cout << "\n==Enter CUDA loop==" << endl;
	
	//FILTERED ARRAY ALLOC
	if (cudaMalloc(&d_filtered_array, (unfiltered_x*unfiltered_y)*sizeof(unsigned int)) != cudaSuccess){
		cout << "Error allocating filtered array space on device" << endl;
		return 0;
	}

	//PADDED ARRAY ALLOC/COPY
	if (cudaMalloc(&d_padded_array, (padded_arr_size*padded_arr_size)*sizeof(unsigned int)) != cudaSuccess){
		cout << "Error allocating padded_array space on device" << endl;
		cudaFree(d_filtered_array);
		return 0;
	}
	if (cudaMemcpy(d_padded_array, padded_array, (padded_arr_size*padded_arr_size)*sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess){
		cout << "Error copying window array to GPU" << endl;
		//cudaFree(d_window);
		cudaFree(d_filtered_array);
		cudaFree(d_padded_array);
		return 0;
	}

	//ORIGINAL ARRAY ALLOC/COPY
	/*unsigned int* d_unfiltered;
	if (cudaMalloc(&d_unfiltered, (unfiltered_x*unfiltered_x)*sizeof(unsigned int)) != cudaSuccess){
		cout << "Error allocating filtered array space on device" << endl;
		return 0;
	}
	if (cudaMemcpy(d_unfiltered, unfiltered_array, (unfiltered_x*unfiltered_y)*sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess){
		cout << "Error copying window array to GPU" << endl;
		//cudaFree(d_window);
		cudaFree(d_filtered_array);
		cudaFree(d_padded_array);
		return 0;
	}*/
	

	//double startSerial = omp_get_wtime();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
	const dim3 grid((padded_arr_size + TILE_W - 1) / TILE_W, (padded_arr_size + TILE_H - 1) / TILE_H);

	const dim3 grid_no_sm((unfiltered_x + block.x - 1) / block.x, (unfiltered_y + block.y - 1) / block.y );
	
	cout << "Execute CUDA Kernel\n";
	cudaFuncSetCacheConfig(mf_cuda, cudaFuncCachePreferL1);
	cudaEventRecord(start);
	//Original_Kernel_Function_no_shared << <grid, block >> >(d_unfiltered, d_filtered_array, unfiltered_x, unfiltered_y);
	//mf_cuda << <grid_no_sm, block >> >(d_padded_array, padded_arr_size, window_size, d_filtered_array, unfiltered_x, mid);
	mf_cuda_sm<<<grid, block>>>(d_padded_array, padded_arr_size, window_size, d_filtered_array, unfiltered_x, mid);
	//mf_cuda_osm << <grid, block >> >(d_padded_array, d_filtered_array, padded_arr_size, padded_arr_size);
	cudaEventRecord(stop);
	//cout << "HERE!\n";
	cudaDeviceSynchronize();

	//cout << "copy data back" << endl;
	if (cudaMemcpy(result_array, d_filtered_array, (unfiltered_x*unfiltered_y)*sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess){
		cout << "Error copying array back from GPU" << endl;
	}
	//cout << "HERE2!\n";

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "::Loop time CUDA:\t\t" << milliseconds/(float)1000 << endl;

	/*for (int i = 0; i < unfiltered_x; i++){
		for (int j = 0; j < unfiltered_y; j++){
			cout << result_array[unfiltered_x*i + j] << " ";
		}
		cout << endl;
	}*/

	//double endSerial = omp_get_wtime();

	//cout << "::Loop time CUDA:\t\t" << endSerial - startSerial << endl;

	printToFile(bins, result_array, unfiltered_x, unfiltered_y, "filtered_parallel.csv");
	
	//cout << "Free Cuda mem" << endl;
	cudaFree(d_filtered_array);
	cudaFree(d_padded_array);

	//cout << "Delete arrays" << endl;
	//delete[] filtered_array;
	//delete[] padded_array;
	//delete[] result_array;

	return 0;
};