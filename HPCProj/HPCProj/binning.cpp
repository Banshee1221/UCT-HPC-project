#include "stdafx.h"
#include "binning.h"


class Binning{

	private:
		vector<vector<unsigned int>> pointsTemp;
		vector<float> binsTemp;

	void binner(float current_x, float current_y, int x_in, int y_in){
		int counter_x = 0, counter_y = 0;

		for (int i = 0; i < x_in; ++i){
			if (i == 0){
				if (current_x <= binsTemp[i]){
					counter_x = 0;
				}
			}
			else{
				if (current_x <= binsTemp[i] && current_x > binsTemp[i - 1]){
					counter_x = i;
				}
			}
		}

		for (int i = 0; i < y_in; ++i){
			if (i == 0){
				if (current_y <= binsTemp[i]){
					counter_y = 0;
				}
			}
			else{
				if (current_y <= binsTemp[i] && current_y > binsTemp[i - 1]){
					counter_y = i;
				}
			}

		}
		pointsTemp[counter_x][counter_y]++;
	};
};