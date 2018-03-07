#define _CRT_SECURE_NO_WARNINGS
#include "stdafx.h"
#include "lesion_classify.hpp"


void lesion_classify(std::vector<Lesion>& lesions) {
	std::vector<int> lesion_id;
	
	//std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-Senior-Design-Smart-Mirror-Image-Processing/rgb/output/4_lesion_classify/");
	//file stuff for testing
	

	//----------------------testing----------------------------
	/*
	fprintf(pFile, "------MEAN------ \n");
	cv::Mat show_mean(64, 64, CV_32FC3);
	show_mean = cv::Scalar(lesions[0].get_bg_color());
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
	std::string img_out_dir = "C:/Users/Austin Pursley/Desktop/ECEN-Senior-Design-Smart-Mirror-Image-Processing/rgb/output/4_lesion_classify/" + std::to_string(Lesion::img_id) + "/";
	_mkdir(img_out_dir.c_str());
	FILE * pFile;
	std::string text_file = img_out_dir + std::to_string(Lesion::img_id) + "_color_data.txt";
	pFile = fopen(text_file.c_str(), "w");
	for (int i = 0; i < lesions.size();i++) {
		

		cv::Mat les_color_mat(64, 64, CV_32FC3);
		les_color_mat = cv::Scalar(lesions[i].get_color());
		
		cv::Mat les_hsv_mat(64, 64, CV_32FC3);
		cv::cvtColor(les_color_mat, les_hsv_mat, CV_RGB2HSV);
		cv::Vec3f hsv_color = les_hsv_mat.at<cv::Vec3f>(cv::Point(0, 0));

		cv::Mat les_bg_color_mat(64, 64, CV_32FC3);
		les_bg_color_mat = cv::Scalar(lesions[i].get_bg_color());
		
		cv::Mat les_bg_hsv_mat(64, 64, CV_32FC3);
		cv::cvtColor(les_bg_color_mat, les_bg_hsv_mat, CV_RGB2HSV);
		cv::Vec3f bg_hsv_color = les_bg_hsv_mat.at<cv::Vec3f>(cv::Point(0, 0));
		//----------------------testing----------------------------
		/*
		fprintf(pFile, "------lesion%d------ \n", (i));
		fprintf(pFile, "lesion%d size is %.2f \n", (i), lesions[i].get_area());
		cv::imwrite(img_out_dir + "les_color_" + std::to_string(i) + ".jpg", les_bg_color_mat);
		cv::imwrite(img_out_dir + "les_color_" + std::to_string(i) + ".jpg", les_color_mat);
		*/
		//--------------------testing-----------------------------
		double perc_diff_hue = ((hsv_color[0] - bg_hsv_color[0]) / bg_hsv_color[0]) * 100;
		double perc_diff_sat = ((hsv_color[1] - bg_hsv_color[1]) / bg_hsv_color[1]) * 100;
		double perc_diff_val = ((hsv_color[2] - bg_hsv_color[2]) / bg_hsv_color[2]) * 100;
		
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
	//cv::Mat m(1, 1, CV_8UC3);
	//m = cv::Scalar(lesion_colors[0]);
}