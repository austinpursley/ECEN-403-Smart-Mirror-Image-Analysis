#define NOMINMAX
#include "stdafx.h"
#include "../my_in_out_directory.hpp"
#include <opencv2/opencv.hpp>

void mask_features(cv::Mat face, std::map<std::string, cv::Rect> features, cv::Mat& masked_face);
cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height, cv::Scalar paddingColor);
void extract_skin(cv::Mat img, std::map<std::string, cv::Rect> features, cv::Mat& skin);