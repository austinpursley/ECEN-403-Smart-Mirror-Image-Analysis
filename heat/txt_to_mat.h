#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stdafx.h"

cv::Mat txt_to_mat(std::string txt_file, int height, int width) {
	
	cv::Mat mat(height, width, CV_32SC1);
	std::ifstream ifs(txt_file);
	std::string tempstr;
	int tempint;
	int row = 0;
	int column = 0;
	while (std::getline(ifs, tempstr)) {
		column = 0;
		std::istringstream iss(tempstr);
		std::vector<int> tempv;
		while (iss >> tempint) {
			mat.at<int>(row, column) = tempint;
			column++;
		}
		row++;
	}
	return mat;
}
