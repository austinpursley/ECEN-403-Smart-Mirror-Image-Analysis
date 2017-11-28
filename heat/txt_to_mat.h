#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stdafx.h"

cv::Mat txt_to_mat(std::string txt_file, int width, int height) {
	//The data in txt will be integer values, hence CV_32SC1
	cv::Mat mat(height, width, CV_32SC1);
	std::ifstream ifs(txt_file);
	std::string row;
	int tempint;
	int i = 0;
	int j = 0;
	while (std::getline(ifs, row)) {
		j = 0;
		std::istringstream iss(row);
		std::vector<int> tempv;
		while (iss >> tempint) {
			mat.at<int>(i, j) = tempint;
			j++;
		}
		i++;
	}
	return mat;
}
