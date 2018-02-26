#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <map>
#include <stdio.h>

// Function Headers
void get_face_features(cv::Mat frame, cv::CascadeClassifier &face_cascade, cv::CascadeClassifier &eyes_cascade, cv::CascadeClassifier &mouth_cascade, std::map<std::string, cv::Rect>& features);


// Global variables
// Copy this file from opencv/data/haarscascades to target folder
//string window_name = "Capture - Face detection";
//int filenumber; // Number of file to be saved
//string filename;