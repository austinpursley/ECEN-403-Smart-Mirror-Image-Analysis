#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesion_features.h"
#include "stdafx.h"
 
void color_filter(cv::Mat src, const std::vector<std::vector<cv::Point> > & src_contours, std::vector<std::vector<cv::Point>> & dst, std::string img_name, int thresh = 30, double roi_scale = 0.25) {

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
	 
	 int contour_cnt = src_contours.size();
	 int size = src.cols*roi_scale;

	 cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	 cv::Mat pad_mask;
	 cv::drawContours(mask, src_contours, -1, cv::Scalar(255), -1);

	 cv::bitwise_not(mask, mask);
	 cv::copyMakeBorder(mask, pad_mask, size, size, size, size, cv::BORDER_CONSTANT, 0);

	 for (int i = 0; i < contour_cnt;i++) {
		 //local mean color
		 cv::Mat pad_mat;
		 cv::Mat color(64, 64, CV_32FC3);
		 cv::Mat color_scale(64, 64, CV_32FC3);
		 cv::Mat color_hsv(64, 64, CV_32FC3);
		 cv::Mat color_lab(64, 64, CV_32FC3);
		 cv::Vec3f mean_hsv_color;
		 cv::Vec3f mean_lab_color;

		 cv::Rect cnt_roi = cv::boundingRect(src_contours[i]);
		 cv::copyMakeBorder(src, pad_mat, size, size, size, size, cv::BORDER_REPLICATE, 0);
		 cv::Size inflationSize(size * 2, size * 2);
		 cnt_roi += inflationSize;

		 if (cnt_roi.x >= 0 && cnt_roi.y >= 0 && cnt_roi.width + cnt_roi.x < pad_mat.cols && cnt_roi.height + cnt_roi.y < pad_mat.rows) {
			 cv::Mat image_roi = pad_mat(cnt_roi);
			 cv::Mat image_roi_mask = pad_mask(cnt_roi);
			 cv::Scalar mean_color = cv::mean(image_roi, image_roi_mask);
			 cv::Mat mean_color_mat(64, 64, CV_32FC3, mean_color);
			 cv::Mat mean_color_mat_scale = mean_color_mat * 1. / 255;
			 cv::cvtColor(mean_color_mat, color_hsv, CV_RGB2HSV);
			 cv::cvtColor(mean_color_mat_scale, color_lab, CV_BGR2YCrCb);
			 mean_hsv_color = color_hsv.at<cv::Vec3f>(cv::Point(0, 0));
			 mean_lab_color = color_lab.at<cv::Vec3f>(cv::Point(0, 0));
			 
			 cv::imwrite(img_out_dir + std::to_string(i) + "_les_roi" + ".jpg", image_roi);
			 cv::imwrite(img_out_dir + std::to_string(i) + "_mask" + ".jpg", image_roi_mask);
			 
		 }
		//lesion colors
		color = cv::Scalar(les_colors[i + 1]);
		color_scale = color * 1. / 255;
		cv::cvtColor(color, color_hsv, CV_RGB2HSV);
		cv::cvtColor(color_scale, color_lab, CV_BGR2YCrCb);
		cv::Vec3f hsv_color = color_hsv.at<cv::Vec3f>(cv::Point(0, 0));
		cv::Vec3f lab_color = color_lab.at<cv::Vec3f>(cv::Point(0, 0));

		double perc_diff_hue = ((hsv_color[0] - mean_hsv_color[0]) / mean_hsv_color[0]) * 100;
		double perc_diff_sat = std::abs(((hsv_color[1] - mean_hsv_color[1]) / mean_hsv_color[1]) * 100);
		double perc_diff_val = std::abs(((mean_hsv_color[2] - hsv_color[2]) / mean_hsv_color[2]) * 100);
		double perc_diff_A = std::abs(((lab_color[1] - mean_lab_color[1]) / mean_lab_color[1]) * 100);
		double perc_diff_B = std::abs(((lab_color[2] - mean_lab_color[2]) / mean_lab_color[2]) * 100);
		
		///------------- TESTING / DEBUG ---------------------
		fprintf(pFile, "------lesion%d------ \n", (i));
		fprintf(pFile, "------mean------ \n");
		fprintf(pFile, "hue:     %f \n", mean_hsv_color[0]);
		fprintf(pFile, "satur:   %f \n", mean_hsv_color[1]);
		fprintf(pFile, "value:   %f \n", mean_hsv_color[2]);
		fprintf(pFile, "A:   %f \n", mean_lab_color[1]);
		fprintf(pFile, "B:   %f \n\n", mean_lab_color[2]);
		fprintf(pFile, "------les------ \n");
		fprintf(pFile, "hue:     %f \n", hsv_color[0]);
		fprintf(pFile, "satur:   %f \n", hsv_color[1]);
		fprintf(pFile, "value:   %f \n", hsv_color[2]);
		fprintf(pFile, "A:   %f \n", lab_color[1]);
		fprintf(pFile, "B:   %f \n\n", lab_color[2]);
		fprintf(pFile, "perc_diff_hue: %f \n", perc_diff_hue);
		fprintf(pFile, "perc_diff_sat: %f \n", perc_diff_sat);
		fprintf(pFile, "perc_diff_val: %f \n", perc_diff_val);
		fprintf(pFile, "perc_diff_A: %f \n", perc_diff_A);
		fprintf(pFile, "perc_diff_B: %f \n\n", perc_diff_B);
		///--------------------------------------------------- 

		//if (perc_diff_sat > 10.0 && (perc_diff_sat > 5)) {
		//if (perc_diff_A > 10) {
		if (perc_diff_val > 10 || perc_diff_A > 1.0 || perc_diff_B > 2.5) {
		//if (perc_diff_B > 2.5 ) {
			fprintf(pFile, "lesion%d is different enough \n\n", i);
			dst.push_back(src_contours[i]);
			cv::imwrite(img_out_dir + std::to_string(i) + "_les_color" + ".jpg", color);
		}
		else {

			fprintf(pFile, "lesion%d is too similar to skin \n\n", i);
			cv::imwrite(img_out_dir + "NO_" + std::to_string(i) + "_les_color" + ".jpg", color);
			//dst.erase(dst.begin() + (i));
		}
	 }
	 fprintf(pFile, "size of contours now: %f \n\n", dst.size());

	 return;
 }

void filter_lesions_by_entropy(const cv::Mat1b & src_gray, const std::vector<std::vector<cv::Point> > & src_contours, std::vector<std::vector<cv::Point>> & dst, std::string img_name, int thresh = 30, double roi_scale = 0.25) {
	///------------- TESTING / DEBUG ---------------------
	std::string img_out_dir = output_dir + "/entropy_filter/";
	_mkdir(img_out_dir.c_str());
	img_out_dir = img_out_dir + img_name + "/";
	_mkdir(img_out_dir.c_str());

	FILE * pFile;
	std::string out_file = img_out_dir + "/classify_data.txt";
	pFile = fopen(out_file.c_str(), "w");
	///-----------------------------------------------------

	std::vector<double> lesion_entrops;
	cv::Mat entropy_img;
	lesion_entropies(src_gray, src_contours, lesion_entrops, entropy_img);
	
	//getLocalEntropyImage(src_gray, entropy_img);
	//cv::Mat entropy(64, 64, CV_32FC1);
	int size = src_gray.cols * roi_scale;
	printf("size: %d \n", size);

	int contour_cnt = src_contours.size();
	cv::Mat image_roi, image_roi_mask, pad_mat;
	
	double global_entropy_mean = lesion_entrops[0];
	

	cv::imwrite(img_out_dir + "00_enropy_img" + ".jpg", entropy_img);
	

	cv::Mat mask = cv::Mat::zeros(src_gray.rows, src_gray.cols, CV_8UC1);
	cv::Mat pad_mask;
	cv::drawContours(mask, src_contours, -1, cv::Scalar(255), -1);

	cv::bitwise_not(mask, mask);
	cv::copyMakeBorder(mask, pad_mask, size, size, size, size, cv::BORDER_CONSTANT, 0);
	cv::imwrite(img_out_dir + "_les_roi_mask" + ".jpg", mask);

	for (int i = 0; i < contour_cnt; i++) {
		//local entropy values
		cv::Rect cnt_roi = cv::boundingRect(src_contours[i]);
		//int size = cnt_roi.width*local_scale;
		cv::Size inflationSize(size * 2, size * 2);
		cnt_roi += inflationSize;
		cv::Mat g, entropy_mat;
		cv::copyMakeBorder(entropy_img, pad_mat, size, size, size, size, cv::BORDER_REPLICATE, 0);
		if (cnt_roi.x >= 0 && cnt_roi.y >= 0 && cnt_roi.width + cnt_roi.x < pad_mat.cols && cnt_roi.height + cnt_roi.y < pad_mat.rows) {
			image_roi_mask = pad_mask(cnt_roi);
			image_roi = pad_mat(cnt_roi);
			double local_entropy = cv::mean(image_roi, image_roi_mask).val[0];
			double les_entropy = lesion_entrops[i];
			double perc_diff_entropy = std::abs(((les_entropy - local_entropy) / local_entropy) * 100);
			cv::imwrite(img_out_dir + std::to_string(i) + "_les_roi" + ".jpg", image_roi);
			cv::imwrite(img_out_dir + std::to_string(i) + "_les_mask" + ".jpg", image_roi_mask);
			/*
			///------------- TESTING / DEBUG ---------------------
			printf("entropy mean: %f \n", local_entropy);
			printf("entropy: %f \n", les_entropy);
			printf("perc_diff_entropy: %f \n\n", perc_diff_entropy);
			fprintf(pFile, "------lesion%d------ \n", (i));
			fprintf(pFile, "entropy:     %f \n", entropy);
			fprintf(pFile, "perc_diff_entroppy: %f \n\n", perc_diff_entropy);
			///---------------------------------------------------
			*/
			if (perc_diff_entropy > thresh) {
				fprintf(pFile, "lesion%d is different enough \n\n", i);
				dst.push_back(src_contours[i]);
				//cv::imwrite(img_out_dir + std::to_string(i) + "_les_color" + ".jpg", entropy);
			}
			else {
				fprintf(pFile, "lesion%d is too similar to skin \n\n", i);
				//cv::imwrite(img_out_dir + "NO" + std::to_string(i) + "_les_color" + ".jpg", entropy);
			}
			
		}
	}
	
	fprintf(pFile, "size of contours now: %f \n\n", dst.size());

	return;
}