#include <fstream>
#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

cv::Mat txt_to_mat(std::string txt_file, const int width, const int height);

double medianMat(cv::Mat Input, const int nVals);

double temp_from_thermal_img(const cv::Mat &thermal);