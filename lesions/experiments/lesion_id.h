#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "my_in_out_directory.hpp"
#include "stdafx.h"

std::vector<int> lesion_id(std::vector<cv::Scalar> & lesion_colors, std::string img_name) {
	
	///------------- TESTING / DEBUG ---------------------
	std::string img_out_dir = output_dir + "/classification/";
	_mkdir(img_out_dir.c_str());
	img_out_dir = img_out_dir + img_name + "/";
	_mkdir(img_out_dir.c_str());

	FILE * pFile;
	std::string out_file = img_out_dir + "/classify_data.txt";
	pFile = fopen(out_file.c_str(), "w");
	cv::Mat show_mean(64, 64, CV_32FC3);
	show_mean = cv::Scalar(lesion_colors[0]);
	cv::Mat show_hsv(64, 64, CV_32FC3);
	cv::cvtColor(show_mean, show_hsv, CV_RGB2HSV);
	cv::Vec3f mean_hsv_color = show_hsv.at<cv::Vec3f>(cv::Point(0, 0));
	cv::imwrite(img_out_dir + "mean_skin_color.jpg", show_mean);
	fprintf(pFile, "------MEAN------ \n");
	fprintf(pFile, "hue:     %f \n", mean_hsv_color[0]);
	fprintf(pFile, "satur:   %f \n", mean_hsv_color[1]);
	fprintf(pFile, "value:   %f \n\n", mean_hsv_color[2]);
	/*
	//code for YCrCb lab space
	cv::Mat show_cyy(64, 64, CV_32FC3);
	cv::cvtColor(show_mean, show_cyy, CV_BGR2YCrCb);
	cv::Vec3f cyy_color = show_cyy.at<cv::Vec3f>(cv::Point(0, 0));
	fprintf(pFile, "lab0_L:  %f \n", cyy_color[0]);
	fprintf(pFile, "lab1_GR: %f \n", cyy_color[1]);
	fprintf(pFile, "lab2_BY: %f \n\n", cyy_color[2]);
	*/
	///---------------------------------------------------

	std::vector<int> lesion_id;

	for (int i = 1; i < lesion_colors.size();i++) {
			
		cv::Mat show(64, 64, CV_32FC3);
		show = cv::Scalar(lesion_colors[i]);
		cv::Mat show_hsv(64, 64, CV_32FC3);
		cv::cvtColor(show, show_hsv, CV_RGB2HSV);
		cv::Vec3f hsv_color = show_hsv.at<cv::Vec3f>(cv::Point(0, 0));
		
		double perc_diff_hue = ((hsv_color[0] - mean_hsv_color[0]) / mean_hsv_color[0]) * 100;
		double perc_diff_sat = ((hsv_color[1] - mean_hsv_color[1]) / mean_hsv_color[1]) * 100;
		double perc_diff_val = ((mean_hsv_color[2] - hsv_color[2]) / mean_hsv_color[2]) * 100;
		
		///------------- TESTING / DEBUG ---------------------
		cv::imwrite(img_out_dir + std::to_string(i - 1) + "_les_color" + ".jpg", show);
		fprintf(pFile, "------lesion%d------ \n", (i - 1));
		fprintf(pFile, "hue:     %f \n", hsv_color[0]);
		fprintf(pFile, "satur:   %f \n", hsv_color[1]);
		fprintf(pFile, "value:   %f \n", hsv_color[2]);
		fprintf(pFile, "perc_diff_hue: %f \n", perc_diff_hue);
		fprintf(pFile, "perc_diff_sat: %f \n", perc_diff_sat);
		fprintf(pFile, "perc_diff_val: %f \n\n", perc_diff_val);
		///---------------------------------------------------

		if (perc_diff_sat > 12.0 || perc_diff_val > 25.0) {
			if ((perc_diff_hue > 3.25)) {
				if (perc_diff_val < 10.0) {

					fprintf(pFile, "lesion%d is red (1) \n\n", i - 1);
				}
				else if ((perc_diff_sat < 25.0) && (perc_diff_val < 30.0) ) {
					fprintf(pFile, "lesion%d is red (2) \n\n", i - 1);
				}
				else {
					fprintf(pFile, "lesion%d is dark (red) \n\n", i - 1);
				}
			}
			else {
				if (perc_diff_val < 5.0) {
					fprintf(pFile, "lesion%d is red, but light \n\n", i - 1);
				}
				else {
					fprintf(pFile, "lesion%d is dark \n\n", i - 1);
				}
			}
		}
		else {
			fprintf(pFile, "lesion%d is too similar to skin \n\n", i - 1);
		}
	}

	//thinking color matrix is easier to work with
	cv::Mat m(1, 1, CV_8UC3);
	m = cv::Scalar(lesion_colors[0]);
	return  lesion_id;
}