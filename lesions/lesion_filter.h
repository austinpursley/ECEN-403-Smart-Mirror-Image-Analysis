#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesion_features.h"
#include "stdafx.h"
 
void color_filter(cv::Mat src, std::vector<std::vector<cv::Point> > & lesion_contours, std::string img_name, int thresh = 30, double roi_scale = 0.25) {

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
	 std::vector<cv::Scalar> bg_colors;

	 std::vector<std::vector<cv::Point> > filtered_les_cnts;
	 printf( "size of contours before1: %d \n", lesion_contours.size());

	 lesion_colors(src, lesion_contours, les_colors, bg_colors, roi_scale);
	 printf("size of contours before2: %d \n", lesion_contours.size());
	 // minus one to discount average lesion size at  the end of vector
	 int contour_cnt = les_colors.size() - 1;

	 for (int i = 0; i < contour_cnt;i++) {
		 cv::Mat color_hsv(64, 64, CV_32FC3);
		 cv::Mat color_lab(64, 64, CV_32FC3);
		 cv::Mat color_yrb(64, 64, CV_32FC3);

		 //background colors
		 cv::Scalar cnt_bg_color = bg_colors[i];
		 cv::Mat bg_color_mat(64, 64, CV_32FC3, cnt_bg_color);
		 cv::Mat bg_color_mat_scale = bg_color_mat * 1. / 255;
		 
		 cv::cvtColor(bg_color_mat, color_hsv, CV_RGB2HSV);
		 cv::Vec3f bg_hsv_color = color_hsv.at<cv::Vec3f>(cv::Point(0, 0));
		 cv::cvtColor(bg_color_mat_scale, color_lab, CV_BGR2Lab);
		 cv::Vec3f bg_lab_color = color_lab.at<cv::Vec3f>(cv::Point(0, 0));
		 cv::cvtColor(bg_color_mat_scale, color_yrb, CV_BGR2YCrCb);
		 cv::Vec3f bg_yrb_color = color_yrb.at<cv::Vec3f>(cv::Point(0, 0));

		//lesion colors
		cv::Mat color(64, 64, CV_32FC3);
		cv::Mat color_scale(64, 64, CV_32FC3);
		color = cv::Scalar(les_colors[i]);
		color_scale = color * 1. / 255;

		cv::cvtColor(color, color_hsv, CV_RGB2HSV);
		cv::Vec3f hsv_color = color_hsv.at<cv::Vec3f>(cv::Point(0, 0));
		cv::cvtColor(color_scale, color_lab, CV_BGR2Lab);
		cv::Vec3f lab_color = color_lab.at<cv::Vec3f>(cv::Point(0, 0));
		cv::cvtColor(color_scale, color_yrb, CV_RGB2YCrCb);
		cv::Vec3f yrb_color = color_yrb.at<cv::Vec3f>(cv::Point(0, 0));

		double perc_diff_hue = ((hsv_color[0] - bg_hsv_color[0]) / bg_hsv_color[0]) * 100;
		//double perc_diff_sat = std::abs(((hsv_color[1] - bg_hsv_color[1]) / bg_hsv_color[1]) * 100);
		//double perc_diff_val = std::abs(((bg_hsv_color[2] - hsv_color[2]) / bg_hsv_color[2]) * 100);
		//double perc_diff_A = std::abs(((lab_color[1] - bg_lab_color[1]) / bg_lab_color[1]) * 100);
		//double perc_diff_B = std::abs(((lab_color[2] - bg_lab_color[2]) / bg_lab_color[2]) * 100);

		double perc_diff_sat = (((hsv_color[1] - bg_hsv_color[1]) / bg_hsv_color[1]) * 100);
		double perc_diff_val = (((bg_hsv_color[2] - hsv_color[2]) / bg_hsv_color[2]) * 100);
		double perc_diff_A = (((lab_color[1] - bg_lab_color[1]) / bg_lab_color[1]) * 100);
		double perc_diff_B = (((lab_color[2] - bg_lab_color[2]) / bg_lab_color[2]) * 100);
		double perc_diff_Cr = (((yrb_color[1] - bg_yrb_color[1]) / bg_yrb_color[1]) * 100);
		double perc_diff_Cb = (((yrb_color[2] - bg_yrb_color[2]) / bg_yrb_color[2]) * 100);
		
		 

		if (perc_diff_val > 10 && perc_diff_Cr < -8 && perc_diff_Cb > 5) {
			///------------- TESTING / DEBUG ---------------------
			fprintf(pFile, "------lesion id: %d------ \n", (i));
			fprintf(pFile, "------bg col------ \n");
			fprintf(pFile, "hue:     %f \n", bg_hsv_color[0]);
			//fprintf(pFile, "satur:   %f \n", bg_hsv_color[1]);
			//fprintf(pFile, "value:   %f \n", bg_hsv_color[2]);
			//fprintf(pFile, "A:   %f \n", bg_lab_color[1]);
			fprintf(pFile, "B:   %f \n\n", bg_lab_color[2]);
			fprintf(pFile, "------les col------ \n");
			fprintf(pFile, "hue:     %f \n", hsv_color[0]);
			//fprintf(pFile, "satur:   %f \n", hsv_color[1]);
			//fprintf(pFile, "value:   %f \n", hsv_color[2]);
			//fprintf(pFile, "A:   %f \n", lab_color[1]);
			fprintf(pFile, "B:   %f \n\n", lab_color[2]);

			fprintf(pFile, "perc_diff_hue: %f \n", perc_diff_hue);
			fprintf(pFile, "perc_diff_sat: %f \n", perc_diff_sat);
			fprintf(pFile, "perc_diff_val: %f \n", perc_diff_val);
			fprintf(pFile, "perc_diff_A: %f \n", perc_diff_A);
			fprintf(pFile, "perc_diff_B: %f \n", perc_diff_B);
			fprintf(pFile, "perc_diff_Cr: %f \n", perc_diff_Cr);
			fprintf(pFile, "perc_diff_Cb: %f \n\n", perc_diff_Cb);
			///---------------------------------------------------

			//darkest moles
			if (perc_diff_val > 30 || (perc_diff_sat > 30 && perc_diff_val > 20)) {
				fprintf(pFile, "suppose to be darkest mole\n");
				fprintf(pFile, "lesion%d is different enough \n\n", i);
				filtered_les_cnts.push_back(lesion_contours[i]);
			}
			else if (perc_diff_val > 19) {
				if (std::abs(perc_diff_A) > 30 || std::abs(perc_diff_B) > 30) {
					fprintf(pFile, "suppose to be dark mole\n");
					fprintf(pFile, "lesion%d is different enough \n\n", i);
					filtered_les_cnts.push_back(lesion_contours[i]);
				}
			}
			else if (perc_diff_val > 10) {
				if (perc_diff_A < 5 || (perc_diff_B < -10 && perc_diff_B > -25)) {
					fprintf(pFile, "maybe light mole\n");
					fprintf(pFile, "lesion%d is different enough \n\n", i);
					filtered_les_cnts.push_back(lesion_contours[i]);
				}
			}
			/*
			if (perc_diff_val > 20 && (std::abs(perc_diff_A) > 27 || std::abs(perc_diff_B) > 50)) {
				fprintf(pFile, "suppose to be dark mole\n");
				fprintf(pFile, "lesion%d is different enough \n\n", i);
				filtered_les_cnts.push_back(lesion_contours[i]);
			}
			*/

			/*
			//brown birth mark white skin
			if ((perc_diff_val > 30 || (perc_diff_val > 20 && perc_diff_sat > 20)) && (abs(perc_diff_A) > 30 || abs(perc_diff_B) > 30)) {
			fprintf(pFile, "suppose to be brown birth mark, white skin \n");
			fprintf(pFile, "lesion%d is different enough \n\n", i);
			filtered_les_cnts.push_back(lesion_contours[i]);
			}
			//dark regions on darker / black skin
			else if (std::abs(perc_diff_B) > 30 && perc_diff_Cr < -8 && perc_diff_A < -10 && perc_diff_val > 25){
				fprintf(pFile, "suppose to be darker regions on darker / black skin \n");
				fprintf(pFile, "lesion%d is different enough \n\n", i);
				filtered_les_cnts.push_back(lesion_contours[i]);
			}
			//brown moles
			else if (perc_diff_Cr < -25 && perc_diff_Cb > 50 && perc_diff_sat > 0 && perc_diff_A < 95 && perc_diff_B < 0) {
				fprintf(pFile, "suppose to be brown moles \n");
				fprintf(pFile, "lesion%d is different enough \n\n", i);
				filtered_les_cnts.push_back(lesion_contours[i]);
				//cv::imwrite(img_out_dir + std::to_string(i) + "_les_color" + ".jpg", color);
			}
			*/
		}
		else {
			fprintf(pFile, "lesion%d is too similar to skin \n\n", i);
			//cv::imwrite(img_out_dir + "NO_" + std::to_string(i) + "_les_color" + ".jpg", color);
			//dst.erase(dst.begin() + (i));
		}
	 
		
	 
	 
	 
	 }

	 lesion_contours = filtered_les_cnts;

	 printf("size of filtered contours now: %d \n", filtered_les_cnts.size());
	 //printf("size of contours now: %d \n\n", lesion_contours.size());

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
	
	double global_entropy_bg = lesion_entrops[0];
	

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
			printf("entropy bg: %f \n", local_entropy);
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