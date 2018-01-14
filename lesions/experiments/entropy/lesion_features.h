#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "fix_me_lesions.h"
#include "lesion_entropy_detect.h"
#include "stdafx.h"

///to do: get rid of return (more like other opencv function). Use this to get local mean color as well (instead of global mean)
void lesion_colors(const cv::Mat & image, std::vector<std::vector<cv::Point> > & contours, std::vector<cv::Scalar> & lesion_colors, std::vector<cv::Scalar> & background_colors, const double roi_scale = 0.25) {
	
	///------------------ PRESENTATION / DEBUG ---------------
	std::string img_out_dir = output_dir + "/colors/";
	_mkdir(img_out_dir.c_str());
	//img_out_dir = img_out_dir + img_name + "/";
	img_out_dir = img_out_dir + "/";
	_mkdir(img_out_dir.c_str());
	///-------------------------------------------------------

	//cv::Mat mask;
	//calculate the mean color of entire image
	
	double size = image.cols*roi_scale;

	std::vector<std::vector<cv::Point> > copycontours;

	cv::Mat all_mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	cv::Mat pad_mask;
	cv::drawContours(all_mask, contours, -1, cv::Scalar(255), -1);
	cv::bitwise_not(all_mask, all_mask);
	cv::copyMakeBorder(all_mask, pad_mask, size, size, size, size, cv::BORDER_CONSTANT, 0);
	int lesion_num = 0;
	for (int i = 0; i < contours.size(); i++) {
		
		cv::Mat pad_mat;
		cv::Rect cnt_roi = cv::boundingRect(contours[i]);
		cv::copyMakeBorder(image, pad_mat, size, size, size, size, cv::BORDER_REPLICATE, 0);
		cv::Size inflationSize(size * 2, size * 2);
		cnt_roi += inflationSize;
		
		//should make this a seperate function, filter lesion that do not pass this test
		if (cnt_roi.x >= 0 && cnt_roi.y >= 0 && cnt_roi.width + cnt_roi.x < pad_mat.cols && cnt_roi.height + cnt_roi.y < pad_mat.rows) {
			
			copycontours.push_back(contours[i]);
			//lesion color
			cv::Mat single_mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
			cv::drawContours(single_mask, contours, i, cv::Scalar(255), -1);
			cv::Scalar les_color = cv::mean(image, single_mask);
			lesion_colors.push_back(les_color);
			//background color
			cv::Mat image_roi = pad_mat(cnt_roi);
			cv::Mat image_roi_mask = pad_mask(cnt_roi);
			cv::Scalar bg_color = cv::mean(image_roi, image_roi_mask);
			background_colors.push_back(bg_color);

			///------------------ PRESENTATION / DEBUG ---------------
			cv::Mat show = image.clone();
			cv::drawContours(show, contours, i, cv::Scalar(255), 1, 1);
			//cv::imwrite(img_out_dir + std::to_string(lesion_num) + "_les_roi" + ".jpg", image_roi);
			//cv::imwrite(img_out_dir + std::to_string(lesion_num) + "_mask" + ".jpg", image_roi_mask);
			cv::imwrite(img_out_dir + std::to_string(lesion_num) + "_les" + ".jpg", show);
			lesion_num++;
			///--------------------------------------------------------
		}
		else {
			
			printf("SKIPPED LESION: %D \n", i);
			continue;
		}
		
	}
	contours = copycontours;
	//at the end of the vectors, put average global background color and avg color of all lesions.
	cv::Scalar average_les_color = cv::mean(image, all_mask);
	lesion_colors.push_back(average_les_color);
	
	cv::bitwise_not(all_mask, all_mask);
	cv::Scalar global_bg_color = cv::mean(image, all_mask);
	background_colors.push_back(global_bg_color);

	return;
}

//Input array of contours/lesions, grayscale. Output array of entropy values corresponding to lesions, as well as image of entropy image.
void lesion_entropies(const cv::Mat &gray, const std::vector<std::vector<cv::Point> > &contours, std::vector<double> &lesion_entrops, cv::Mat &entropy_mat) {
	cv::Mat g(gray);
	cv::Mat dst = cv::Mat::zeros(g.rows, g.cols, CV_32F);

	entropy_filter(g, dst);
	cv::normalize(dst, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_32FC1);
	dst.convertTo(entropy_mat, CV_8U);
	
	//calculate the mean entropy of entire image
	//this is first element of output vector
	double mean_entropy = cv::mean(entropy_mat).val[0];
	lesion_entrops.push_back(mean_entropy);

	for (int i = 0; i < contours.size(); i++) {
		//mask is a black image
		cv::Mat mask = cv::Mat::zeros(entropy_mat.rows, entropy_mat.cols, CV_8UC1);
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(mask, contours, i, cv::Scalar(255), -1);
		//mask means we are only considering white contour section
		double les_entropy = cv::mean(entropy_mat, mask).val[0];
		
		lesion_entrops.push_back(les_entropy);

		///------------------ PRESENTATION / DEBUG ---------------
		//printf("les_entropy: %f \n", lesion_entrops[i]);
		std::string img_out_dir = output_dir + "/lesion_entropies/";
		_mkdir(img_out_dir.c_str());
		//img_out_dir = img_out_dir + img_name + "/";
		img_out_dir = img_out_dir + "/";
		_mkdir(img_out_dir.c_str());

		cv::Mat show;
		entropy_mat.copyTo(show);
		cv::drawContours(show, contours, i, cv::Scalar(0), 1, 1);
		cv::imwrite(img_out_dir + std::to_string(i) + "_les" + ".jpg", show);
		
		///--------------------------------------------------------
	}
	return;
}


 std::vector<double> lesion_areas(const std::vector<std::vector<cv::Point> > & contours) {
	 //each lesion has an associated contour and here we calculate the area of contours
	 std::vector<double> contour_areas;
	 double area;
	 for (int i = 0; i < contours.size(); i++) {
		 area = cv::contourArea(contours[i]);
		 contour_areas.push_back(area);
	 }
	 return(contour_areas);
 }


