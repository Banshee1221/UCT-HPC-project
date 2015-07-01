// HPCProj.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
	std::ifstream input_file;
	input_file.open("Points_[1.0e+08]_Noise_[030]_Normal.bin");
	float x_coord;
	float y_coord;
	int count = 0;
	float point[2];
	do {
		count++;
		input_file.read((char *)point, 8);
		std::cout << "x: " << point[0] << " y: " << point[1] << " pos: " << input_file.tellg() << " eof: " << input_file.eof() << std::endl;
	} while (!input_file.eof());
	std::cout << "count: " << count - 1 << std::endl;
}

