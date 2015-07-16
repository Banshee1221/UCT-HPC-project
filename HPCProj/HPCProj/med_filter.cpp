#include "stdafx.h"
#include "med_filter.h"
#include <omp.h>

vector<vector<unsigned int>> array_padder(int original_size, int window_size, vector<vector<unsigned  int>> original_array){
	int pad_val = (window_size - 1) / 2;
	vector<vector<unsigned int>> padded_array;
	padded_array.resize(original_size + window_size - 1);

	int padded_size = padded_array.size();

	for (int i = 0; i < padded_size; ++i){
		padded_array[i].resize(original_size + window_size - 1);
	}
	//cout << "got here" << endl;
	for (int i = pad_val; i < padded_size - pad_val; ++i){
		for (int j = pad_val; j < padded_size - pad_val; ++j){
			padded_array[i][j] = original_array[i - pad_val][j - pad_val];	
		}
	}

	//Symmetry
	for (int x = pad_val; x < pad_val + original_size; ++x){
		int count = 0;
		for (int top = pad_val - 1; top >= 0; --top){
			padded_array[top][x] = original_array[count][x - pad_val];
			padded_array[x][top] = original_array[x - pad_val][count];
			count++;
		}
	}

	for (int y = original_size - 1; y >= 0; --y){
		for (int space = 0; space < pad_val; space++){
			padded_array[y + pad_val][padded_size - pad_val + space] = original_array[y][original_size - 1 - space];
		}
	}

	int tmpVal = original_size - 1;
	for (int y = original_size + pad_val; y < padded_size; ++y){
		for (int x = 0 + pad_val; x < original_size + pad_val; ++x){
			padded_array[y][x] = original_array[tmpVal][x - pad_val];
		}
		tmpVal--;
	}
	//EndSymmetry

	/*for (int i = 0; i < padded_size; i++){
		for (int j = 0; j < padded_size; j++){
			if (padded_array[i][j] == 0){
				cout << padded_array[i][j] << "\t";
			}
			else
				cout << padded_array[i][j] << "\t";
		}
		cout << endl;
	}*/

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