#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include "writer.h"
#include "med_filter.h"

using namespace std;
using namespace std::chrono;


int main(int argc, char* argv[]) {

	string in_file_name = argv[1];
	int x_in = atoi(argv[2]); 
	int y_in = x_in;
	cout << "Input - x: " << x_in << " y: " << y_in << endl;

	ifstream input_file;
	input_file.open(in_file_name, ios::binary);
	
	float x_min = 0, x_max = 1, y_min = 0, y_max = 1;
	int count = 0;
	
	float current_x, current_y;

	cout << "start" << endl;
	cout << "create bins array" << endl;
	vector<float> bins;
	bins.resize(x_in);
	for (int i = 0; i < x_in; i++){
		bins[i] = ((float)(i + 1) / (float)x_in);
	}
	cout << "bins array: " << endl;
	for (int i = 0; i < x_in; i++){
		cout << bins[i] << endl;
	}

	cout << "create 2d" << endl;
	//---
	vector<vector<unsigned int>> array_points;
	array_points.resize(x_in);
	for (int i = 0; i < x_in; ++i){
		array_points[i].resize(y_in);
	}
	
	//---
	cout << "2d array fill " << endl;
	for (int i = 0; i < x_in; i++){
		for (int j = 0; j < y_in; j++)
			array_points[i][j] = 0;
			//cout << "x: " << bins[i] << " | y: " << bins[j] << "\t" << array_points[i][j] << "\n";
	}

	/*cout << "2d array: " << endl;
	for (int i = 0; i < x_in; i++){
		for (int j = 0; j < y_in; j++)
		cout << "x: " << bins[i] << " | y: " << bins[j] << "\t" << array_points[i][j] << "\n";
	}*/

	int counter_x = 0, counter_y = 0;
	high_resolution_clock::time_point loop_t1 = high_resolution_clock::now();
	do {
		input_file.read(reinterpret_cast<char*>(&current_x), sizeof(float));
		input_file.read(reinterpret_cast<char*>(&current_y), sizeof(float));		
		for (int i = 0; i < x_in; ++i){
			if (i == 0){
				if (current_x <= bins[i]){
					counter_x = 0;
				}
			}
			else{
				if (current_x <= bins[i] && current_x > bins[i-1]){
					counter_x = i;
				}
			}
		}

		for (int i = 0; i < y_in; ++i){
			if (i == 0){
				if (current_y <= bins[i]){
					counter_y = 0;
				}
			}
			else{
				if (current_y <= bins[i] && current_y > bins[i - 1]){
					counter_y = i;
				}
			}
			
		}
		array_points[counter_x][counter_y]++;
	} while (!input_file.eof());
	high_resolution_clock::time_point loop_t2 = high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(loop_t2 - loop_t1).count();
	cout << "Loop time: " << duration/(float)1000000 << endl; 
	cout << "Here" << endl;
	
	//Formatting
	printToFile(bins, array_points, x_in, y_in, "unfiltered.csv");


	if (argc == 4){

	

		int x_res = atoi(argv[3]);
		int y_res = x_res;

		medianFilter(bins, x_res, array_points, x_in, y_in);
		
		return 0;
	}


	
	

	/*for (int i = 0; i < x_in; ++i)
		for (int j = 0; j < y_in; ++j)
			cout << "x: " << bins[i] << " | y: " << bins[j] << "\t" << array_points[i][j] << "\n";*/
	

	 /*cout << "deleting 2d arr points" << endl;
	 for (int i = 0; i < x_in; i++){
		 delete[] array_points[i];
	 }
	 cout << "deleting 2d arr" << endl;
	 delete[] array_points;*/

	//cout << "count: " << (count - 1)/2 << endl;
	//cout << "min_x: " << x_min << endl;
	//cout << "max_x: " << x_max << endl;
	//cout << "min_y: " << y_min << endl;
	//cout << "max_y: " << y_max << endl;
	 return 0;
};