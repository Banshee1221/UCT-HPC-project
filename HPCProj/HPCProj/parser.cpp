#include "stdafx.h"
#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdio>
#include "writer.h"
#include "med_filter.h"

#define THREADS 7
#define READ_BUFFER 10000000
#define PARALLEL
//#define SERIAL

using namespace std;

int main(int argc, char* argv[]) {

	// Set OMP threads
	omp_set_num_threads(THREADS);

	// Get input from command line arguments
	int x_res = atoi(argv[3]);
	int y_res = x_res;
	if (argc == 4){
		if (x_res < 3 || x_res > 21 || x_res % 2 == 0){
			cout << "Incorrect filter size!\nSize must be between 3 and 21.\n";
			return 0;
		}
	}

#ifdef PARALLEL
	cout << "=== RUNNING IN PARALLEL MODE ===\n" << endl;
#endif

	string in_file_name = argv[1];
	int x_in = atoi(argv[2]);
	int y_in = x_in;

	if (x_in < 5 || x_in > 4096){
		cout << "Incorrect bin size!\nSize must be between 5 and 4096.\n";
		return 0;
	}
	if (x_res >= x_in){
		cout << "Incorrect filter size!\nSize must be smaller than the bin size.\n";
		return 0;
	}

	const int gridSize = x_in * y_in;
	cout << "Binning size:\t\t\t" << x_in << " x " << y_in << endl;
	if (argc == 4){
		cout << "Filter Window:\t\t\t" << x_res << " x " << y_res << endl;
	}

	// Open file for reading
	ifstream input_file;
	input_file.open(in_file_name, ios::binary | ios::in | ios::ate);
	unsigned __int64 length = input_file.tellg();
	cout << "Input File Name:\t\t" << argv[1] << endl << "Buffer size:\t\t\t" << READ_BUFFER << " Bytes"
		<< endl;
#ifdef PARALLEL
	cout << "Threads specified:\t\t" << THREADS << endl;
#endif
	cout << endl;
	input_file.seekg(ios::beg);

	cout << "Start binning process\n=====================" << endl;

	// Create bins array and populate
	cout << "::Create bins array" << endl;
	vector<float> bins;
	bins.resize(x_in);
	for (int i = 0; i < x_in; i++){
		bins[i] = ((float)(i + 1) / (float)x_in);
	}

	// Create coordinate array and initialize
	cout << "::Create points array" << endl;

	unsigned __int64 *array_points;
	array_points = new (nothrow) unsigned __int64[gridSize]();

	// Set up temporary variables for parsing purposes
	float current_x, current_y;
	int counter_x = -1, counter_y = -1;
	int count = 0;
	vector<float> buffer;
	buffer.resize(READ_BUFFER);
	double start = 0;
	double end = 0;

#ifdef SERIAL
	cout << "\n==Enter serial loop== " << endl;
	
	while (!input_file.eof() && !input_file.fail()){

		if (!(input_file.read(reinterpret_cast<char*>(&buffer[0]), READ_BUFFER*sizeof(float)))){
			break;
		}
		start = omp_get_wtime();
		for (int i = 0; i < (buffer.size() - 1); i += 2){

			current_x = buffer[i];
			current_y = buffer[i + 1];

			for (int a = 0; a < x_in; ++a){
				if (a == 0){
					if (current_x < bins[0]){
						counter_x = 0;
					}
					if (current_y < bins[0]){
						counter_y = 0;
					}
				}
				else{
					if (current_x < bins[a] && current_x >= bins[a - 1]){
						counter_x = a;
					}
					if (current_y < bins[a] && current_y >= bins[a - 1]){
						counter_y = a;
					}
				}
			}
			array_points[counter_y*y_in + counter_x]++;
			count++;
		}
		end += (omp_get_wtime() - start);

	}

	cout << "::Coordinates processed:\t" << count << endl;
	cout << "::Loop time serial:\t\t" << end << endl;
	cout << "\n=>Writing serial data to file" << endl;

	printToFile(bins, array_points, x_in, y_in, "unfiltered_serial.csv");

#endif

#ifdef PARALLEL

	count = 0;

	buffer.clear();
	buffer.resize(READ_BUFFER);

	input_file.close();
	input_file.open(in_file_name, ios::binary | ios::in);

	for (int n = 0; n < gridSize; n++)
		array_points[n] = 0;

	counter_x = -1, counter_y = -1;
	count = 0;
	cout << "\n==Entering OMP loop==" << endl;
	start = 0;
	end = 0;

	while (!input_file.eof() && !input_file.fail()){
		counter_x = -1, counter_y = -1;
		if (!(input_file.read(reinterpret_cast<char*>(&buffer[0]), READ_BUFFER*sizeof(float)))){
			break;
		}
		start = omp_get_wtime();
#pragma omp parallel for private(counter_x, counter_y) schedule(static)
			for (int i = 0; i < (buffer.size() - 1); i += 2){

				current_x = buffer[i];
				current_y = buffer[i + 1];

				for (int a = 0; a < x_in; ++a){
					if (a == 0){
						if (current_x < bins[a]){
							counter_x = 0;
						}
						if (current_y < bins[a]){
							counter_y = 0;
						}
					}
					else{
						if (current_x < bins[a] && current_x >= bins[a - 1]){
							counter_x = a;
						}
						if (current_y < bins[a] && current_y >= bins[a - 1]){
							counter_y = a;
						}
					}
				}
#pragma omp atomic
				array_points[counter_y*y_in + counter_x]++;
#pragma omp atomic
				count++;
			}
			end += (omp_get_wtime() - start);
		}

	cout << "::Coordinates processed:\t" << count << endl;
	cout << "::Loop time parallel:\t\t" << end << endl;
	cout << "\n=>Writing parallel data to file" << endl;
	printToFile(bins, array_points, x_in, y_in, "unfiltered_parallel.csv");
#endif

	if (argc == 4){

		cout << "\nStarting median filter process\n==============================\n";
#ifdef SERIAL	
		medianFilter(bins, x_res, array_points, x_in, y_in);
#endif
		
#ifdef PARALLEL
		medianFilter(bins, x_res, array_points, x_in, y_in);
		printf("\n\nCUDA\n\n");
		medianFilter_CUDA(bins, x_res, array_points, x_in, y_in);
#endif
	}

	delete[] array_points;
	return 0;
};