#include <iostream>
#include <fstream>
#include <vector>

#ifndef WRITER_H
#define WRITER_H

using namespace std;

void printToFile(vector<float> bins, vector<vector<unsigned int>> counted_array, int square_width, int square_height, string output_file_name);

#endif