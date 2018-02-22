#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "stdafx.h"

std::vector<int> lesion_id(std::vector<cv::Scalar> & lesion_colors, std::vector<double> lesion_area) {
	std::vector<int> lesion_id;
	
	//std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/output/");
	
	//file stuff for testing
	FILE * pFile;
	pFile = fopen ("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/lesions/output/color_data.txt", "w");

	//----------------------testing----------------------------
	fprintf(pFile, "------MEAN------ \n");
	cv::Mat show_mean(64, 64, CV_32FC3);
	show_mean = cv::Scalar(lesion_colors[0]);
	cv::imwrite(image_out + "mean_skin_color.jpg", show_mean);

	cv::Mat show_hsv(64, 64, CV_32FC3);
	cv::cvtColor(show_mean, show_hsv, CV_RGB2HSV);
	
	cv::Vec3f mean_hsv_color = show_hsv.at<cv::Vec3f>(cv::Point(0, 0));
	fprintf(pFile, "hue:     %f \n", mean_hsv_color[0]);
	fprintf(pFile, "satur:   %f \n", mean_hsv_color[1]);
	fprintf(pFile, "value:   %f \n\n", mean_hsv_color[2]);

	/*
	cv::Mat show_cyy(64, 64, CV_32FC3);
	cv::cvtColor(show_mean, show_cyy, CV_BGR2YCrCb);
	cv::Vec3f cyy_color = show_cyy.at<cv::Vec3f>(cv::Point(0, 0));
	fprintf(pFile, "lab0_L:  %f \n", cyy_color[0]);
	fprintf(pFile, "lab1_GR: %f \n", cyy_color[1]);
	fprintf(pFile, "lab2_BY: %f \n\n", cyy_color[2]);
	*/

	//--------------------testing-----------------------------

	for (int i = 1; i < lesion_colors.size();i++) {
		
		//----------------------testing----------------------------
		fprintf(pFile, "------lesion%d------ \n", (i-1));
		fprintf(pFile, "lesion%d size is %.2f \n", (i-1), lesion_area[i-1]);
		cv::Mat show(64, 64, CV_32FC3);
		show = cv::Scalar(lesion_colors[i]);
		cv::imwrite(image_out + "les_color_" + std::to_string(i-1) + ".jpg", show);

		cv::Mat show_hsv(64, 64, CV_32FC3);
		cv::cvtColor(show, show_hsv, CV_RGB2HSV);

		cv::Vec3f hsv_color = show_hsv.at<cv::Vec3f>(cv::Point(0, 0));
		
		//--------------------testing-----------------------------
		double perc_diff_hue = ((hsv_color[0] - mean_hsv_color[0]) / mean_hsv_color[0]) * 100;
		double perc_diff_sat = ((hsv_color[1] - mean_hsv_color[1]) / mean_hsv_color[1]) * 100;
		double perc_diff_val = ((mean_hsv_color[2] - hsv_color[2]) / mean_hsv_color[2]) * 100;
		
		fprintf(pFile, "hue:     %f \n", hsv_color[0]);
		fprintf(pFile, "satur:   %f \n", hsv_color[1]);
		fprintf(pFile, "value:   %f \n", hsv_color[2]);
		fprintf(pFile, "perc_diff_hue: %f \n", perc_diff_hue);
		fprintf(pFile, "perc_diff_sat: %f \n", perc_diff_sat);
		fprintf(pFile, "perc_diff_val: %f \n\n", perc_diff_val);
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