#include "stdafx.h"
#include "med_filter.h"
#include <omp.h>

vector<vector<unsigned int>> array_padder(int original_size, int window_size, vector<vector<unsigned  int>> original_array){
	vector<vector<unsigned int>> padded_array;
	padded_array.resize(original_size + window_size - 1);

	for (int i = 0; i < padded_array.size(); ++i){
		padded_array[i].resize(original_size + window_size - 1);
	}
	//cout << "got here" << endl;
	for (int i = (window_size - 1) / 2; i < padded_array.size() - (window_size - 1) / 2; ++i){
		for (int j = (window_size - 1) / 2; j < padded_array.size() - (window_size - 1) / 2; ++j){
			padded_array[i][j] = original_array[i - (window_size - 1) / 2][j - (window_size - 1) / 2];	
		}
	}

	for (int i = 0; i < padded_array.size(); i++){
		for (int j = 0; j < padded_array.size(); j++){
			if (padded_array[i][j] == 0){
				cout << padded_array[i][j] << "\t";
			}
			else
				cout << padded_array[i][j] << "\t";
		}
		cout << endl;
	}

	return padded_array;
}

void medianFilter(vector<float> bins, int window_size, vector<vector<unsigned int>> unfiltered_array, int unfiltered_x, int unfiltered_y){
	//int y, x, yy, xx, p, yyy, xxx, mid = (window_size - 1) / 2;
	int p, mid = (window_size - 1) / 2;

	cout << "::Create window array" << endl;
	vector<float> window;
	window.resize(window_size * window_size);

	cout << "::Create 2D filtered array" << endl;
	vector<vector<unsigned int>> filtered_points;
	filtered_points.resize(unfiltered_x);
	for (int i = 0; i < unfiltered_x; ++i){
		filtered_points[i].resize(unfiltered_y);
	}

	for (int i = 0; i < unfiltered_x; i++){
		for (int j = 0; j < unfiltered_y; j++)
			filtered_points[i][j] = 0;
	}

	vector<vector<unsigned int>> padded_array = array_padder(unfiltered_x, window_size, unfiltered_array);

	cout << "\n==Enter serial loop==" << endl;
	double startSerial = omp_get_wtime();
	
	for (int column = (window_size - 1) / 2; column < padded_array.size() - (window_size - 1) / 2; column++)
	{
		for (int row = (window_size - 1) / 2; row < padded_array.size() - (window_size - 1) / 2; row++)
		{
			p = 0;
			for (int c = column - (window_size - 1) / 2; c <= column + (window_size - 1) / 2; c++)
				for (int r = row - (window_size - 1) / 2; r <= row + (window_size - 1) / 2; r++)
				{
					window[p] = padded_array[c][r];
					p++;
				}
			sort(window.begin(), window.end());
			filtered_points[column - (window_size - 1) / 2][row - (window_size - 1) / 2] = window[(window_size*window_size) / 2];
				
		}
	}

	double endSerial = omp_get_wtime();

	cout << "::Loop time serial:\t\t" << endSerial - startSerial << endl;

	printToFile(bins, filtered_points, unfiltered_x, unfiltered_y, "filtered_serial.csv");
};