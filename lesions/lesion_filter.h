#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesion_features.h"
#include "stdafx.h"
 
void lesion_filter(cv::Mat src, const std::vector<std::vector<cv::Point> > & src_contours, std::vector<std::vector<cv::Point>> & dst, std::string img_name) {

	 ///------------- TESTING / DEBUG ---------------------
	 std::string img_out_dir = output_dir + "/color_filter/";
	 _mkdir(img_out_dir.c_str());
	 img_out_dir = img_out_dir + img_name + "/";
	 _mkdir(img_out_dir.c_str());

	 FILE * pFile;
	 std::string out_file = img_out_dir + "/classify_data.txt";
	 pFile = fopen(out_file.c_str(), "w");
	 ///-----------------------------------------------------

	 std::vector<cv::Scalar> les_colors;
	 les_colors = lesion_colors(src, src_contours);
	 
	 cv::Mat color_mean(64, 64, CV_32FC3, cv::Scalar(les_colors[0]));
	 cv::Mat color_hsv(64, 64, CV_32FC3);
	 cv::Mat color_lab(64, 64, CV_32FC3);

	cv::Mat color_mean_scale = color_mean * 1. / 255;

	 cv::cvtColor(color_mean, color_hsv, CV_RGB2HSV);
	 cv::cvtColor(color_mean_scale, color_lab, CV_BGR2Lab);

	 cv::imwrite(img_out_dir + "mean_skin_color.jpg", color_mean);

	 cv::Vec3f mean_hsv_color = color_hsv.at<cv::Vec3f>(cv::Point(0, 0));
	 cv::Vec3f mean_lab_color = color_lab.at<cv::Vec3f>(cv::Point(0, 0));

	 int  contour_cnt = src_contours.size();
	 cv::Mat color(64, 64, CV_32FC3);
	 cv::Mat color_scale(64, 64, CV_32FC3);
	 ///------------- TESTING / DEBUG ---------------------
	 fprintf(pFile, "------mean------ \n");
	 fprintf(pFile, "hue:     %f \n", mean_hsv_color[0]);
	 fprintf(pFile, "satur:   %f \n", mean_hsv_color[1]);
	 fprintf(pFile, "value:   %f \n", mean_hsv_color[2]);
	 fprintf(pFile, "A:   %f \n", mean_lab_color[1]);
	 fprintf(pFile, "B:   %f \n\n", mean_lab_color[2]);
	 ///---------------------------------------------------

	 for (int i = 0; i < contour_cnt;i++) {

		 color = cv::Scalar(les_colors[i+1]);
		 color_scale = color * 1. / 255;
		 cv::cvtColor(color, color_hsv, CV_RGB2HSV);
		 cv::cvtColor(color_scale, color_lab, CV_BGR2Lab);
		 cv::Vec3f hsv_color = color_hsv.at<cv::Vec3f>(cv::Point(0, 0));
		 cv::Vec3f lab_color = color_lab.at<cv::Vec3f>(cv::Point(0, 0));

		 double perc_diff_hue = ((hsv_color[0] - mean_hsv_color[0]) / mean_hsv_color[0]) * 100;
		 double perc_diff_sat = std::abs(((hsv_color[1] - mean_hsv_color[1]) / mean_hsv_color[1]) * 100);
		 double perc_diff_val = ((mean_hsv_color[2] - hsv_color[2]) / mean_hsv_color[2]) * 100;
		 double perc_diff_A = ((lab_color[1] - mean_lab_color[1]) / mean_lab_color[1]) * 100;
		 double perc_diff_B = ((lab_color[2] - mean_lab_color[2]) / mean_lab_color[2]) * 100;
		 ///------------- TESTING / DEBUG ---------------------
		 
		 fprintf(pFile, "------lesion%d------ \n", (i));
		 fprintf(pFile, "hue:     %f \n", hsv_color[0]);
		 fprintf(pFile, "satur:   %f \n", hsv_color[1]);
		 fprintf(pFile, "value:   %f \n", hsv_color[2]);
		 fprintf(pFile, "A:   %f \n", lab_color[1]);
		 fprintf(pFile, "B:   %f \n", lab_color[2]);
		 fprintf(pFile, "perc_diff_hue: %f \n", perc_diff_hue);
		 fprintf(pFile, "perc_diff_sat: %f \n", perc_diff_sat);
		 fprintf(pFile, "perc_diff_val: %f \n", perc_diff_val);
		 fprintf(pFile, "perc_diff_A: %f \n", perc_diff_A);
		 fprintf(pFile, "perc_diff_B: %f \n\n", perc_diff_B);
		 ///---------------------------------------------------

		 //if (perc_diff_sat > 10.0 || (perc_diff_val > 60.0 && perc_diff_sat > 0)) {
		 if (perc_diff_A > 20) {
			 fprintf(pFile, "lesion%d is different enough \n\n", i);
			 dst.push_back(src_contours[i]);
			 cv::imwrite(img_out_dir + std::to_string(i) + "_les_color" + ".jpg", color);
		 }
		 else {

			 fprintf(pFile, "lesion%d is too similar to skin \n\n", i);
			 cv::imwrite(img_out_dir + "NO" + std::to_string(i) + "_les_color" + ".jpg", color);
			 //dst.erase(dst.begin() + (i));
		 }
	 }
	 fprintf(pFile, "size of contours now: %f \n\n", dst.size());

	 return;
 }

void entropy_filter(cv::Mat1b src_gray, const std::vector<std::vector<cv::Point> > & src_contours, std::vector<std::vector<cv::Point>> & dst, std::string img_name) {

	///------------- TESTING / DEBUG ---------------------
	std::string img_out_dir = output_dir + "/entropy_filter/";
	_mkdir(img_out_dir.c_str());
	img_out_dir = img_out_dir + img_name + "/";
	_mkdir(img_out_dir.c_str());

	FILE * pFile;
	std::string out_file = img_out_dir + "/classify_data.txt";
	pFile = fopen(out_file.c_str(), "w");
	///-----------------------------------------------------

	std::vector<cv::Scalar> les_entropy;
	les_entropy = lesion_entropies(src_gray, src_contours);

	double entropy_mean = (cv::Scalar(les_entropy[0])).val[0];

	int  contour_cnt = src_contours.size();
	//cv::Mat entropy(64, 64, CV_32FC1);
	double entropy;
	double entropy_scale;
	///------------- TESTING / DEBUG ---------------------
	fprintf(pFile, "------mean------ \n");
	fprintf(pFile, "entropy_mean:     %f \n", entropy_mean);
	///---------------------------------------------------

	for (int i = 0; i < contour_cnt;i++) {

		entropy = (cv::Scalar(les_entropy[i + 1])).val[0];
		//entropy_scale = entropy * 1. / 255;

		double perc_diff_entropy = ((entropy - entropy_mean) / entropy_mean) * 100;
		///------------- TESTING / DEBUG ---------------------

		fprintf(pFile, "------lesion%d------ \n", (i));
		fprintf(pFile, "entropy:     %f \n", entropy);
		fprintf(pFile, "perc_diff_entroppy: %f \n\n", perc_diff_entropy);
		///---------------------------------------------------

		if (perc_diff_entropy > 0) {
			fprintf(pFile, "lesion%d is different enough \n\n", i);
			dst.push_back(src_contours[i]);
			//cv::imwrite(img_out_dir + std::to_string(i) + "_les_color" + ".jpg", entropy);
		}
		else {

			fprintf(pFile, "lesion%d is too similar to skin \n\n", i);
			//cv::imwrite(img_out_dir + "NO" + std::to_string(i) + "_les_color" + ".jpg", entropy);
			//dst.erase(dst.begin() + (i));
		}
	}
	fprintf(pFile, "size of contours now: %f \n\n", dst.size());

	return;
}