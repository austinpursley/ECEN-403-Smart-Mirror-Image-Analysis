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
	 
	 cv::Mat show_mean(64, 64, CV_32FC3);
	 show_mean = cv::Scalar(les_colors[0]);

	 cv::Mat show_hsv(64, 64, CV_32FC3);
	 cv::cvtColor(show_mean, show_hsv, CV_RGB2HSV);

	 cv::imwrite(img_out_dir + "mean_skin_color.jpg", show_mean);

	 cv::Vec3f mean_hsv_color = show_hsv.at<cv::Vec3f>(cv::Point(0, 0));
	 cv::Vec3f hsv_color = show_hsv.at<cv::Vec3f>(cv::Point(0, 0));
	 int  contour_cnt = src_contours.size();
	 for (int i = 0; i < contour_cnt;i++) {

		 cv::Mat show(64, 64, CV_32FC3);
		 show = cv::Scalar(les_colors[i+1]);
		 cv::Mat show_hsv(64, 64, CV_32FC3);
		 cv::cvtColor(show, show_hsv, CV_RGB2HSV);
		 cv::Vec3f hsv_color = show_hsv.at<cv::Vec3f>(cv::Point(0, 0));

		 double perc_diff_hue = ((hsv_color[0] - mean_hsv_color[0]) / mean_hsv_color[0]) * 100;
		 double perc_diff_sat = ((hsv_color[1] - mean_hsv_color[1]) / mean_hsv_color[1]) * 100;
		 double perc_diff_val = ((mean_hsv_color[2] - hsv_color[2]) / mean_hsv_color[2]) * 100;

		 ///------------- TESTING / DEBUG ---------------------
		 
		 fprintf(pFile, "------lesion%d------ \n", (i));
		 fprintf(pFile, "hue:     %f \n", hsv_color[0]);
		 fprintf(pFile, "satur:   %f \n", hsv_color[1]);
		 fprintf(pFile, "value:   %f \n", hsv_color[2]);
		 fprintf(pFile, "perc_diff_hue: %f \n", perc_diff_hue);
		 fprintf(pFile, "perc_diff_sat: %f \n", perc_diff_sat);
		 fprintf(pFile, "perc_diff_val: %f \n\n", perc_diff_val);
		 ///---------------------------------------------------

		 if (perc_diff_sat > 10.0 || perc_diff_val > 5.0) {
			 fprintf(pFile, "lesion%d is different enough \n\n", i);
			 dst.push_back(src_contours[i]);
			 cv::imwrite(img_out_dir + std::to_string(i) + "_les_color" + ".jpg", show);
		 }
		 else {

			 fprintf(pFile, "lesion%d is too similar to skin \n\n", i);
			 cv::imwrite(img_out_dir + "NO" + std::to_string(i) + "_les_color" + ".jpg", show);
			 //dst.erase(dst.begin() + (i));
			 contour_cnt--;
		 }
	 }
	 fprintf(pFile, "size of contours now: %f \n\n", dst.size());

	 return;
 }