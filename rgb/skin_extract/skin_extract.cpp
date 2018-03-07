#include "stdafx.h"
#include "skin_extract.h"

void mask_features(cv::Mat face, std::map<std::string, cv::Rect> features, cv::Mat& masked_face) {
	//resize(face, face, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR);
	face.copyTo(masked_face);
	double escale = 0.4;
	cv::Point center(features["eye1"].x + features["eye1"].width / 2, features["eye1"].y + features["eye1"].height / 2);
	cv::Size axes(features["eye1"].height * escale, features["eye1"].width * escale);
	ellipse(masked_face, center, axes, 90, 0, 360, 0, -1);

	center = cv::Point(features["eye2"].x + features["eye2"].width / 2, features["eye2"].y + features["eye2"].height / 2);
	axes = cv::Size(features["eye2"].height * escale, features["eye2"].width * escale);
	ellipse(masked_face, center, axes, 90, 0, 360, 0, -1);

	double mscale = 0.5;
	center = cv::Point(features["mouth"].x + features["mouth"].width / 2, features["mouth"].y + features["mouth"].height / 2);
	axes = cv::Size(features["mouth"].height * mscale, features["mouth"].width * mscale);
	ellipse(masked_face, center, axes, 90, 0, 360, 0, -1);

	return;
}

cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height, cv::Scalar paddingColor) {
	int bottom_right_x = top_left_x + width;
	int bottom_right_y = top_left_y + height;
	
	cv::Mat output;
	if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {

	}
	return output;
}

void skin_morph(cv::Mat& mask) {
	int shape = cv::MORPH_ELLIPSE;
	cv::bitwise_not(mask, mask);

	//int size = 5;
	//cv::Mat elem = cv::getStructuringElement(shape,
	//	cv::Size(2 * size + 1, 2 * size + 1),
	//	cv::Point(size, size));

	//cv::morphologyEx(mask, mask, cv::MORPH_OPEN, elem);

	for (int size = 3; size < 8; size++) {
		cv::Mat elem = cv::getStructuringElement(shape,
			cv::Size(2 * size + 1, 2 * size + 1),
			cv::Point(size, size));
		cv::morphologyEx(mask, mask, cv::MORPH_OPEN, elem);
		cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, elem);
		
	}

	//cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, elem);
	cv::bitwise_not(mask, mask);
}

void extract_skin(cv::Mat img, std::map<std::string, cv::Rect> features, cv::Mat& skin) {
	//TO-DO: black out outside ROI, morphology, mask eyes and mouth
	cv::Mat face = img(features["face"]);
	cv::Mat face_masked;
	mask_features(face, features, face_masked);
	cv::Mat lab_image;
	//cvtColor(matImage, lab_image, cv::COLOR_BGR2Lab);
	cvtColor(face, lab_image, cv::COLOR_RGB2HSV);
	// crop
	double scalex = 0.175;
	double scaley = 0.15;
	int offset_x = face_masked.size().width*scalex;
	int offset_y = face_masked.size().height*scaley;
	cv::Rect roi;
	roi.x = offset_x;
	roi.y = offset_y;
	roi.width = face_masked.size().width - (offset_x * 2);
	roi.height = face_masked.size().height - (offset_y * 2.5);
	cv::Mat crop = face_masked(roi);

	// thresholding and mask
	int thresholdType = cv::THRESH_BINARY;
	int otsu = thresholdType + cv::THRESH_OTSU;
	int otsu_max_value = 255;
	cv::Mat thresh, gray;
	cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
	//cv::threshold(gray, thresh, 0, otsu_max_value, thresholdType + cv::THRESH_OTSU);
	cv::threshold(gray, thresh, 1, 255, cv::THRESH_BINARY);

	//extract mean
	cvtColor(crop, crop, cv::COLOR_RGB2HSV);
	cv::Scalar mean = cv::mean(crop, thresh);
	//cv::Scalar mean = cv::mean(crop);

	//cv::Scalar lower = 0.93*mean;
	//cv::Scalar upper = 1.03*mean;

	cv::Scalar lower = mean;
	cv::Scalar upper = mean;

	lower[0] = lower[0] - 10;
	upper[0] = upper[0] + 10;

	lower[1] = lower[1] - 40;
	upper[1] = upper[1] + 50;

	lower[2] = lower[2] - 40;
	upper[2] = upper[2] + 100;

	//in range
	cv::Mat range;
	cv::inRange(lab_image, lower, upper, range);

	//morphology
	skin_morph(range);
	mask_features(range, features, range);
	

	cvtColor(crop, crop, cv::COLOR_HSV2RGB);
	cv::Mat crop_mask;
	range.copyTo(skin);
	//face.copyTo(skin, range);
	//img.copyTo(skin, range);
}