#include "stdafx.h"
#include <string>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include "color_face.h"
//#include "color_face.h"
#define _CRT_SECURE_NO_DEPRECATE


int main(int argc, char** argv) {
	std::string input_dir("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_face/input/");
	std::string dir_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_face/output/");
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

	double maxL = 0;
	double maxA = 0;
	double maxB = 0;

	double minL = 255;
	double minA = 255;
	double minB = 255;

	for (int i = 2; i < filenames.size(); i++) {
		//read image
		std::string img_file = filenames[i]; //e.g. name.jpg
		size_t lastindex = img_file.find_last_of(".");
		std::string img_name = img_file.substr(0, lastindex); //e.g name (no .jpg extension)

		std::string img_dir(input_dir.c_str() + img_file);
		cv::Mat matImage = cv::imread(img_dir, cv::IMREAD_COLOR);
		if (!matImage.data) {
			std::cout << "Unable to open the file: " << img_dir;
			return 1;
		}
		// thresholding settings
		int thresholdType = cv::THRESH_BINARY;
		int otsu = thresholdType + cv::THRESH_OTSU;
		int otsu_max_value = 255;
		// guassian Blur setting
		cv::Size ksize;
		ksize.height = 5;
		ksize.width = ksize.height;
		cv::Mat blurr_img;
		cv::GaussianBlur(matImage, blurr_img, ksize, 0);

		//convert to lab color space, split channels
		cv::Mat lab_image;
		cvtColor(blurr_img, lab_image, cv::COLOR_BGR2Lab);
		cv::Mat1b lab[3];
		split(lab_image, lab);

		///red-green
		//otsu threshold 1
		

		//otsu threshold 2
		//cv::Mat1b gr_thresh2;
		//threshold_with_mask(lab[1], gr_thresh2, 0, otsu_max_value, thresholdType + cv::THRESH_OTSU, gr_thresh);
		//bitwise_and(gr_thresh, gr_thresh2, gr_thresh2);
		//cv::imwrite(dir_out + "2.2_greenred_thresh.jpg", gr_thresh2);

		///blue-yellow
		//otsu threshold
		
			
		//otsu threshold 2
		//cv::Mat1b by_thresh2;
		//threshold_with_mask(lab[2], by_thresh2, 0, otsu_max_value, thresholdType + cv::THRESH_OTSU, by_thresh);
		//bitwise_and(by_thresh, by_thresh2, by_thresh2);
		//cv::imwrite(dir_out + "1.2_blueyellow_thresh.jpg", by_thresh2);
		
		cv::Mat mat_img_lab;
		cvtColor(matImage, mat_img_lab, CV_BGR2Lab);
		
		cv::Mat test;
		int min_L = 89;
		int min_a = 134;
		int min_b = 138;
		int max_L = 152;
		int max_a = 149;
		int max_b = 163;
		cv::inRange(mat_img_lab, cv::Scalar(0, 0, 0), cv::Scalar(255, 155, 200), test);
		//cv::inRange(mat_img_lab, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), test);
		cv::bitwise_not(test, test);
		test = test * cv::Mat(test.rows, test.cols, CV_8U, cv::mean(lab[1]));
		lab[1] = lab[1] - test;
		cv::imwrite(dir_out + img_name + "_inrange_test.jpg", test);
		

		//cv::Mat gr_thresh;
		//cv::threshold(lab[1], gr_thresh, 0, otsu_max_value, thresholdType + cv::THRESH_OTSU);
		//cv::imwrite(dir_out + img_name + "_2.1_greenred_thresh.jpg", gr_thresh);
		cv::Mat1b gr_thresh2;
		threshold_with_mask(lab[1], gr_thresh2, 0, otsu_max_value, thresholdType + cv::THRESH_OTSU, test);
		bitwise_and(test, gr_thresh2, gr_thresh2);
		cv::imwrite(dir_out + img_name + "_1.1_greenred_thresh.jpg", gr_thresh2);
		cv::imwrite(dir_out + "_1.0_greenred.jpg", lab[1]);

		cv::Mat1b by_thresh;
		cv::threshold(lab[2], by_thresh, 0, otsu_max_value, thresholdType + cv::THRESH_OTSU);
		cv::imwrite(dir_out + img_name + "_2.2_blueyellow_thresh.jpg", by_thresh);
		cv::imwrite(dir_out + img_name + "_2.0_blueyellow.jpg", lab[2]);
		
		cv::Mat mask;
		bitwise_and(gr_thresh2, by_thresh, mask);
		//mask to show where we are extracting mean color
		cv::Mat masked;
		matImage.copyTo(masked, mask);
		cv::imwrite(dir_out + img_name + "_masked.jpg", masked);

		//extract mean
		cv::Scalar mean = cv::mean(mat_img_lab, mask);
		
	
		



		//printf("blue: %f \n", mean[0]);
		//printf("green: %f \n", mean[1]);
		//printf("red: %f \n\n", mean[2]);
		/// TESTING FOR MAX/MIN VALUES
		cv::Mat1b m_lab[3];
		split(mat_img_lab, m_lab);

		double maxL_c;
		double maxA_c;
		double maxB_c;

		double minL_c;
		double minA_c;
		double minB_c;

		cv::Point loc;

		cv::minMaxLoc(m_lab[0], &minL_c, &maxL_c, &loc, &loc, mask); //find minimum and maximum intensities
		cv::minMaxLoc(m_lab[1], &minA_c, &maxA_c, &loc, &loc, mask);
		cv::minMaxLoc(m_lab[2], &minB_c, &maxB_c, &loc, &loc, mask);
		//max
		if (maxL_c > maxL) {
			maxL = maxL_c;
		}
		if (maxA_c > maxA) {
			maxA = maxA_c;
		}
		if (maxB_c  > maxB) {
			maxB = maxB_c;
		}
		//min
		if (minL_c < minL) {
			minL = minL_c;
		}
		if (minA_c < minA) {
			minA = minA_c;
		}
		if (minB_c < minB) {
			minB = minB_c;
		}
		///---------------------------------------
		
		//mean color square
		mean = cv::mean(matImage, mask);
		int tile_size = 64;
		cv::Rect rect(0, 0, tile_size, tile_size);
		cv::Mat ret = cv::Mat(tile_size, tile_size, CV_8UC3, cv::Scalar(0));
		cv::rectangle(ret, rect, mean, CV_FILLED);
		cv::imwrite(dir_out + img_name + "_mean_color.jpg", ret);
	}

	printf("maxL: %f \n", maxL);
	printf("maxA: %f \n", maxA);
	printf("maxB: %f \n\n", maxB);

	printf("minL: %f \n", minL);
	printf("minA: %f \n", minA);
	printf("minB: %f \n\n", minB);
	
	return 0;
}