#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {

	int x_in = atoi(argv[1]); 
	int y_in = atoi(argv[2]);
	cout << "Input - x: " << x_in << " y: " << y_in << endl;

	ifstream input_file;
	ofstream out_file;
	input_file.open("Points_[1.0e+08]_Noise_[030]_Normal.bin", ios::binary);
	out_file.open("data.csv");
	
	float x_min = 0, x_max = 1, y_min = 0, y_max = 1;
	int count = 0;
	
	float current_x, current_y;

	cout << "start" << endl;
	cout << "create bins array" << endl;
	float* bins = new float[x_in];
	for (int i = 0; i < x_in; i++){
		bins[i] = ((float)(i + 1) / (float)x_in);
	}
	cout << "bins array: " << endl;
	for (int i = 0; i < x_in; i++){
		cout << bins[i] << endl;
	}

	cout << "create 2d" << endl;
	//---
	vector<vector<int>> array_points;
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
	out_file << "\t,";
	for (int i = 0; i < x_in; ++i){
		cout << "bins i: " << i << endl;
		if (i != (x_in - 1)){
			if (i == 0)
				out_file << bins[i] / (float)2 << ",";
			else
				out_file << (bins[i - 1] + bins[i]) / (float)2 << ",";
		}
		else
			out_file << (bins[i - 1] + bins[i]) / (float)2;
	}
	out_file << endl;
	cout << "Writing 2d array" << endl;
	 for (int i = 0; i < x_in; ++i){
		 if (i == 0)
			 out_file << bins[i] / (float)2 << ",";
		 else
			 out_file << (bins[i - 1] + bins[i]) / (float)2 << ",";
		for (int j = 0; j < y_in; ++j){
			if (j != (y_in - 1))
				out_file << array_points[j][i] << ",";
			else
				out_file << array_points[j][i];
		}
		out_file << endl;
	}

	 cout << "closing file" << endl;
	 out_file.close();

	/*for (int i = 0; i < x_in; ++i)
		for (int j = 0; j < y_in; ++j)
			cout << "x: " << bins[i] << " | y: " << bins[j] << "\t" << array_points[i][j] << "\n";*/
	
	 cout << "deleting bins" << endl;
	 delete[] bins;
	 bins = NULL;

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
}