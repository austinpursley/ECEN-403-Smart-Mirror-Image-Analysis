#include "stdafx.h"
#define _CRT_SECURE_NO_DEPRECATE
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "my_in_out_directory.hpp"
#include "lesion_localization.h"
#include "dirent.h"


int main(int argc, char** argv) {
	
	//get list of images in input direcotry
	DIR *dpdf;
	struct dirent *epdf;
	std::vector<std::string> filenames;
	dpdf = opendir(input_dir.c_str());
	if (dpdf != NULL) {
		while (epdf = readdir(dpdf)) {
			filenames.push_back(std::string(epdf->d_name));
		}
	}

	//for each image in input directory
	for (int i = 2; i < filenames.size(); i++) {
		
		std::string img_file = filenames[i]; //e.g. name.jpg
		//we pass image name in functions below for the testing
		size_t lastindex = img_file.find_last_of(".");
		std::string img_name = img_file.substr(0, lastindex); //e.g. name
		
		//read image
		std::string img_dir(input_dir.c_str() + img_file);
		cv::Mat matImage = cv::imread(img_dir, cv::IMREAD_COLOR);
		if (!matImage.data) {
			std::cout << "Unable to open the file: " << img_dir;
			return 1;
		}

		//detect lesions
		std::vector<std::vector<cv::Point> > lesion_contours;
		lesion_contours = lesion_localization(matImage, 0, img_name);

	}
	
	return 0;
}
