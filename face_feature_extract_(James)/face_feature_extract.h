#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <map>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void get_face_features(Mat frame, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade, extern CascadeClassifier &mouth_cascade, std::vector<Rect>& features);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
//string window_name = "Capture - Face detection";
//int filenumber; // Number of file to be saved
//string filename;