#include "stdafx.h"
#include "writer.h"

int printToFile(vector<float> bins, unsigned int* counted_array, int square_width, int square_height, string output_file_name){
	cout << "Open output file" << endl;
	ofstream outFile;
	outFile.open(output_file_name);

	outFile << "\t,";

	for (int i = 0; i < square_width; ++i){
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

	cout << "Writing 2D array" << endl;

	for (int i = 0; i < square_width; ++i){
		if (i == 0)
			outFile << bins[i] / (float)2 << ",";
		else
			outFile << (bins[i - 1] + bins[i]) / (float)2 << ",";
		for (int j = 0; j < square_height; ++j){
			if (j != (square_height - 1))
				outFile << counted_array[square_width*i+j] << ",";
			else
				outFile << counted_array[square_width*i + j];
		}
		outFile << endl;
	}

	cout << "Closing file" << endl;
	outFile.close();

	return 0;
};