#include <vector>
#include <algorithm>
#include "writer.h"

#ifndef MEDFILTER_H
#define MEDFILTER_H

void medianFilter(vector<float> bins, int window_size, unsigned int* unfiltered_array, int unfiltered_x, int unfiltered_y);

int medianFilter_CUDA(vector<float> bins, int window_size, unsigned int* unfiltered_array, int unfiltered_x, int unfiltered_y);

#endif