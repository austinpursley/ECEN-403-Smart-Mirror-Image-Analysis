#include "stdafx.h"
#include "lesion_localization.h"

void blob_detect(const cv::Mat1b &src_1b, cv::Mat1b &bin_mask, std::vector<std::vector<cv::Point>> &contours_output) {
	///VARIABLES / SETTINGS
	//mix tuning/performance parameters in one place.
	int gauss_ksize = 15;
	int blocksize = 39;
	int size_close = 0;
	int size_open = 4;
	int size_close2 = 0;
	int size_open2 = 0;
	int size_erode = 1;

	//guassian blur
	cv::Size ksize;
	ksize.height = gauss_ksize;
	ksize.width = ksize.height;
	//morphology
	int shape = cv::MORPH_ELLIPSE;
	cv::Mat elem_close = cv::getStructuringElement(shape,
		cv::Size(2 * size_close + 1, 2 * size_close + 1),
		cv::Point(size_close, size_close));
	cv::Mat elem_open = cv::getStructuringElement(shape,
		cv::Size(2 * size_open + 1, 2 * size_open + 1),
		cv::Point(size_open, size_open));

	cv::Mat elem_open2 = cv::getStructuringElement(shape,
		cv::Size(2 * size_open2 + 1, 2 * size_open2 + 1),
		cv::Point(size_open2, size_open2));
	cv::Mat elem_close2 = cv::getStructuringElement(shape,
		cv::Size(2 * size_close2 + 1, 2 * size_close2 + 1),
		cv::Point(size_close2, size_close2));

	cv::Mat elem_erode = cv::getStructuringElement(shape,
		cv::Size(2 * size_erode + 1, 2 * size_erode + 1),
		cv::Point(size_erode, size_erode));
	cv::Mat gr_img, lab_img, mix_img, gray_img, blur_img, bin_img, close_img, open_img, morph;

	///PROCESS
	/*
	1: guassian blur filter to reduce image noise
	2: adaptive thresholding, binarization
	3: close to fill in gaps
	4: open removes smaller blobs
	5: dilate what's left to make them more prominent
	6: find contours, the points that make up border of area on the original image
	*/

	cv::GaussianBlur(src_1b, blur_img, ksize, 0);
	cv::adaptiveThreshold(blur_img, bin_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, blocksize, 2);
	cv::morphologyEx(bin_img, close_img, cv::MORPH_CLOSE, elem_close);
	cv::morphologyEx(close_img, open_img, cv::MORPH_OPEN, elem_open);
	cv::morphologyEx(open_img, close_img, cv::MORPH_CLOSE, elem_close2);
	cv::morphologyEx(close_img, open_img, cv::MORPH_OPEN, elem_open2);
	cv::morphologyEx(open_img, bin_mask, cv::MORPH_ERODE, elem_erode);
	cv::findContours(bin_mask, contours_output, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	///OUTPUT / DEBUG
	/*
	std::string img_out_dir = output_dir + "/blob_detect/";
	cv::imwrite(img_out_dir + img_name + "_1_src1b_" + ".jpg", src_1b);
	cv::imwrite(img_out_dir + img_name + "_2_blur_" + ".jpg", blur_img);
	cv::imwrite(img_out_dir + img_name + "_3_thresh_" +  ".jpg", thresh_img);
	cv::imwrite(img_out_dir + img_name + "_4_morph_" + ".jpg",   morph);

	*/
	return;
}

void lesion_area_filter(std::vector<Lesion > &lesions, const double min_area, const double max_area) {

	std::vector<Lesion> new_lesions;
	for (int i = 0; i < lesions.size(); i++) {
		double crnt_les_area = lesions[i].get_area();
		if (crnt_les_area < min_area || crnt_les_area > max_area) {
			continue;
		}
		else {
			new_lesions.push_back(lesions[i]);
		}
	}
	lesions = new_lesions;
}

//adapted from SimpleBlobDetector.cpp OpenCV source
//info: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
void lesion_intertia_filter(std::vector<Lesion > &lesions, const double min_intertia_ratio, const double max_intertia_ratio) {
	std::vector<Lesion> new_lesions;
	for (int i = 0; i < lesions.size(); i++) {
		double ratio = lesions[i].get_inertia_ratio();

		if (ratio < min_intertia_ratio || ratio >= max_intertia_ratio) {
			continue;
		}
		else {
			new_lesions.push_back(lesions[i]);
		}
	}
	lesions = new_lesions;
}

void lesion_draw_contours(const std::vector<Lesion > &lesions, cv::Mat &img) {
	std::vector<std::vector<cv::Point> > lesion_contours;
	for (int i = 0; i < lesions.size(); i++) {
		lesion_contours.push_back(lesions[i].get_contour());
	}
	cv::drawContours(img, lesion_contours, -1, cv::Scalar(255), -1);
}

void mask_image(const cv::Mat &mask, cv::Mat &masked_out) {
	cv::Mat mask_copy;
	mask.copyTo(mask_copy);
	cv::Mat color;
	cv::bitwise_not(mask_copy, mask_copy);
	cv::cvtColor(mask_copy, color, CV_GRAY2BGR);
	cv::bitwise_and(color, masked_out, masked_out);
}

std::vector<std::vector<cv::Point>> lesion_localization(const cv::Mat & image, int type, std::string img_name) {

	cv::Mat mix_img;
	if (type == 0) {
		//by value, all dark lesions on light background
		cv::Mat gr_img = image & cv::Scalar(0, 255, 255);
		gr_img = image & cv::Scalar(0, 255, 255);
		cv::Mat lab_img = cv::Mat(image.rows, image.cols, CV_8UC3);
		std::vector<cv::Mat1b> lab(3);
		cv::cvtColor(gr_img, lab_img, CV_BGR2Lab, 3);
		cv::split(lab_img, lab);
		cv::Mat AB;
		cv::addWeighted(lab[1], 0.2, lab[2], 0.8, 0, AB);
		cv::addWeighted(AB, 0.4, lab[0], 0.6, 0, mix_img);
	}
	else if (type == 1) {
		/*
		//redder lesions
		cv::Mat3b yrb_img(image.rows, image.cols, CV_8UC3);
		cv::cvtColor(gr_img, yrb_img, CV_BGR2YCrCb, 3);
		std::vector<cv::Mat1b> yrb(3);
		cv::split(yrb_img, yrb);
		cv::bitwise_not(yrb[1], yrb[1]);
		cv::bitwise_not(lab[0], lab[0]);
		cv::addWeighted(yrb[1], 0.8, lab[0], 0.2, 0, red_mix);
		///
		*/
	}
	else if (type == 2) {
		/*
		cv::Mat3b hsv_img(image.rows, image.cols, CV_8UC3);
		cv::cvtColor(image, hsv_img, CV_BGR2HSV, 3);
		std::vector<cv::Mat1b> hsv(3);
		cv::split(hsv_img, hsv);
		*/
	}
	else {
		printf("defaulted case! \n");
	}

	//blob detection on our mixed, single channel image
	cv::Mat1b bin_mask;
	std::vector<std::vector<cv::Point>> les_contours;
	blob_detect(mix_img, bin_mask, les_contours);

	//Lesion class, stores properties of lesion like color and area
	std::vector<Lesion> lesions;
	for (int i = 0; i < les_contours.size(); i++) {
		;
		int id_num = i;
		lesions.push_back(Lesion(les_contours[i], image, bin_mask, id_num));
		//std::string img_out_dir = output_dir + "/lesion_class/";
		//_mkdir(img_out_dir.c_str());
		//img_out_dir = img_out_dir + img_name + "/";
		//_mkdir(img_out_dir.c_str());
		//cv::Mat show = image.clone();
		//lesions[i].draw(show);
		//cv::imwrite(img_out_dir + std::to_string(i) + "_0les" + ".jpg", show);
	}

	cv::Mat drawn_lesions = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	double min_area = std::sqrt(image.rows * image.cols)*0.05;
	printf("min_area: %f \n", min_area);
	double max_area = 10000;
	lesion_area_filter(lesions, min_area, max_area);
	lesion_intertia_filter(lesions, 0.04);
	lesion_draw_contours(lesions, drawn_lesions);
	///DEBUG -------------------------------------------------
	//FILE * file;
	//std::string out_file = img_out_dir + "/num_of_lesions.txt";
	//file = fopen(out_file.c_str(), "w");
	std::string img_out_dir = output_dir + "/lesion_localization/";
	_mkdir(img_out_dir.c_str());
	//img_out_dir = img_out_dir + img_name + "/";
	//_mkdir(img_out_dir.c_str());
	cv::Mat masked, filter_masked;
	image.copyTo(masked);
	mask_image(bin_mask, masked);
	image.copyTo(filter_masked);
	mask_image(drawn_lesions, filter_masked);
	cv::imwrite(img_out_dir + img_name + "_0_image_" + ".jpg", image);
	//cv::imwrite(img_out_dir + img_name + "_0_bin_mask_" + ".jpg", bin_mask);
	cv::imwrite(img_out_dir + img_name + "_3_masked_" + ".jpg", masked);
	cv::imwrite(img_out_dir + img_name + "_4_filtered_les_" + ".jpg", filter_masked);
	//int num_lesions = contours.size() - 1;
	//fprintf(file, "# lesions: %d \n", num_lesions);
	///--------------------------------------------------------

	return les_contours;
}