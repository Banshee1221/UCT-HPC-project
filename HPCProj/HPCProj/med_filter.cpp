#include "stdafx.h"
#include "med_filter.h"

void medianFilter(vector<float> bins, int window_size, vector<vector<unsigned int>> unfiltered_array, int unfiltered_x, int unfiltered_y){
	int y, x, yy, xx, p, yyy, xxx, mid = (window_size - 1) / 2;

	vector<float> window;
	window.resize(window_size * window_size);

	vector<vector<unsigned int>> filtered_points;
	filtered_points.resize(unfiltered_x);
	for (int i = 0; i < unfiltered_x; ++i){
		filtered_points[i].resize(unfiltered_y);
	}

	for (int i = 0; i < unfiltered_x; i++){
		for (int j = 0; j < unfiltered_y; j++)
			filtered_points[i][j] = 0;
	}

	
	for (int row = 0; row < unfiltered_y; row++)
	{
		for (int column = 0; column < unfiltered_x; column++)
		{
			if (row == 0 || row == unfiltered_y - 1 || column == 0 ||
				column == unfiltered_x - 1)
			{
				filtered_points[row][column] = unfiltered_array[row][column];
				continue;
			}
			p = 0;
			for (int r = row - 1; r < row + 2; r++)
				for (int c = column - 1; c < column + 2; c++)
				{
					window[p++] = unfiltered_array[r][c];
				}
			sort(window.begin(), window.end());
			filtered_points[row][column] = window[(window_size*window_size) / 2];
				
		}
	}

	printToFile(bins, filtered_points, unfiltered_x, unfiltered_y, "filtered.csv");
};