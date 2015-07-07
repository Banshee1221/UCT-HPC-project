#include "stdafx.h"
#include "writer.h"

void printToFile(vector<float> bins, vector<vector<unsigned int>> counted_array, int square_width, int square_height, string output_file_name){
	ofstream outFile;
	outFile.open(output_file_name);

	outFile << "\t,";

	for (int i = 0; i < square_width; ++i){
		cout << "bins i: " << i << endl;
		if (i != (square_width - 1)){
			if (i == 0)
				outFile << bins[i] / (float)2 << ",";
			else
				outFile << (bins[i - 1] + bins[i]) / (float)2 << ",";
		}
		else
			outFile << (bins[i - 1] + bins[i]) / (float)2;
	}

	outFile << endl;

	cout << "Writing 2d array" << endl;

	for (int i = 0; i < square_width; ++i){
		if (i == 0)
			outFile << bins[i] / (float)2 << ",";
		else
			outFile << (bins[i - 1] + bins[i]) / (float)2 << ",";
		for (int j = 0; j < square_height; ++j){
			if (j != (square_height - 1))
				outFile << counted_array[j][i] << ",";
			else
				outFile << counted_array[j][i];
		}
		outFile << endl;
	}

	cout << "closing file" << endl;
	outFile.close();
};