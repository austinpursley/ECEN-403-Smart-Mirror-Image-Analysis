#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include <time.h>
#include <iostream>

double otsu_8u_with_mask(const cv::Mat1b src, const cv::Mat1b& mask)
{
	const int N = 256;
	int M = 0;
	int i, j, h[N] = { 0 };
	for (i = 0; i < src.rows; i++)
	{
		const uchar* psrc = src.ptr(i);
		const uchar* pmask = mask.ptr(i);
		for (j = 0; j < src.cols; j++)
		{
			if (pmask[j])
			{
				h[psrc[j]]++;
				++M;
			}
		}
	}

	double mu = 0, scale = 1. / (M);
	for (i = 0; i < N; i++)
		mu += i*(double)h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < N; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
			continue;

		mu1 = (mu1 + i*p_i) / q1;
		mu2 = (mu - q1*mu1) / q2;
		sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}
	return max_val;
}

double threshold_with_mask(cv::Mat1b& src, cv::Mat1b& dst, double thresh, double maxval, int type, const cv::Mat1b& mask = cv::Mat1b())
{
	if (mask.empty() || (mask.rows == src.rows && mask.cols == src.cols && cv::countNonZero(mask) == src.rows * src.cols))
	{
		// If empty mask, or all-white mask, use cv::threshold
		thresh = cv::threshold(src, dst, thresh, maxval, type);
	}
	else
	{
		// Use mask
		bool use_otsu = (type & cv::THRESH_OTSU) != 0;
		if (use_otsu)
		{
			// If OTSU, get thresh value on mask only
			thresh = otsu_8u_with_mask(src, mask);
			// Remove THRESH_OTSU from type
			type &= cv::THRESH_MASK;
		}

		// Apply cv::threshold on all image
		thresh = cv::threshold(src, dst, thresh, maxval, type);

		// Copy original image on inverted mask
		src.copyTo(dst, ~mask);
	}
	return thresh;
}

///SOURCE: https://stackoverflow.com/questions/20371053/finding-entropy-in-opencv
static int32_t sub_to_ind(int32_t *coords, int32_t *cumprod, int32_t num_dims)
{
	int index = 0;
	int k;

	assert(coords != NULL);
	assert(cumprod != NULL);
	assert(num_dims > 0);

	for (k = 0; k < num_dims; k++)
	{
		index += coords[k] * cumprod[k];
	}

	return index;
}

static void ind_to_sub(int p, int num_dims, const int size[],
	int *cumprod, int *coords)
{
	int j;

	assert(num_dims > 0);
	assert(coords != NULL);
	assert(cumprod != NULL);

	for (j = num_dims - 1; j >= 0; j--)
	{
		coords[j] = p / cumprod[j];
		p = p % cumprod[j];
	}
}

///ADAPTED FROM SOURCE: https://stackoverflow.com/questions/20371053/finding-entropy-in-opencv
void getLocalEntropyImage(cv::Mat &gray, cv::Rect &roi, cv::Mat &entropy)
{
	clock_t func_begin, func_end;
	func_begin = clock();
	//1.define nerghbood model,here it's 9*9
	int neighbood_dim = 2;
	///raising n-size past 9 doesn't seem to help with lesion detection
	int n_size = 9;
	int neighbood_size[] = { n_size, n_size };

	//2.Pad gray_src
	cv::Mat gray_src_mat(gray);
	cv::Mat pad_mat;
	int left = (neighbood_size[0] - 1) / 2;
	int right = left;
	int top = (neighbood_size[1] - 1) / 2;
	int bottom = top;
	cv::copyMakeBorder(gray_src_mat, pad_mat, top, bottom, left, right, cv::BORDER_REPLICATE, 0);
	cv::Mat *pad_src = &pad_mat;
	roi = cv::Rect(roi.x + top, roi.y + left, roi.width, roi.height);

	//3.initial neighbood object,reference to Matlab build-in neighbood object system
	//        int element_num = roi_rect.area();
	//here,implement a histogram by ourself ,each bin calcalate gray value frequence
	int hist_count[256] = { 0 };
	int neighbood_num = 1;
	for (int i = 0; i < neighbood_dim; i++)
		neighbood_num *= neighbood_size[i];

	//neighbood_corrds_array is a neighbors_num-by-neighbood_dim array containing relative offsets
	int *neighbood_corrds_array = (int *)malloc(sizeof(int)*neighbood_num * neighbood_dim);
	//Contains the cumulative product of the image_size array;used in the sub_to_ind and ind_to_sub calculations.
	int *cumprod = (int *)malloc(neighbood_dim * sizeof(*cumprod));
	cumprod[0] = 1;
	for (int i = 1; i < neighbood_dim; i++)
		cumprod[i] = cumprod[i - 1] * neighbood_size[i - 1];
	int *image_cumprod = (int*)malloc(2 * sizeof(*image_cumprod));
	image_cumprod[0] = 1;
	image_cumprod[1] = pad_src->cols;
	//initialize neighbood_corrds_array
	int p;
	int q;
	int *coords;
	for (p = 0; p < neighbood_num; p++) {
		coords = neighbood_corrds_array + p * neighbood_dim;
		ind_to_sub(p, neighbood_dim, neighbood_size, cumprod, coords);
		for (q = 0; q < neighbood_dim; q++)
			coords[q] -= (neighbood_size[q] - 1) / 2;
	}
	//initlalize neighbood_offset in use of neighbood_corrds_array
	int *neighbood_offset = (int *)malloc(sizeof(int) * neighbood_num);
	int *elem;
	for (int i = 0; i < neighbood_num; i++) {
		elem = neighbood_corrds_array + i * neighbood_dim;
		neighbood_offset[i] = sub_to_ind(elem, image_cumprod, 2);
	}

	//4.calculate entroy for pixel
	uchar *array = (uchar *)pad_src->data;
	//here,use entroy_table to avoid frequency log function which cost losts of time
	float entroy_table[82];
	const float log2 = log(2.0f);
	entroy_table[0] = 0.0;
	float frequency = 0;
	for (int i = 1; i < neighbood_num + 1; i++) {
		frequency = (float)i / neighbood_num;
		entroy_table[i] = frequency * (log(frequency) / log2);
	}
	int neighbood_index;
	//int max_index=pad_src->cols*pad_src->rows;
	float e;
	int current_index = 0;
	int current_index_in_origin = 0;
	///changes made here from source, "roi.height + roi.y" and "roi.width + roi.x"
	for (int y = roi.y; y < roi.height + roi.y; y++) {
		current_index = y * pad_src->cols;
		current_index_in_origin = (y - roi.y) * gray.cols;
		for (int x = roi.x; x < roi.width + roi.x; x++, current_index++, current_index_in_origin++) {
			for (int j = 0;j<neighbood_num;j++) {
				neighbood_index = current_index + neighbood_offset[j];
				hist_count[array[neighbood_index]]++;
			}
			//get entropy
			e = 0;
			for (int k = 0; k < 256; k++) {
				if (hist_count[k] != 0) {
					//int frequency=hist_count[k];
					e -= entroy_table[hist_count[k]];
					hist_count[k] = 0;
				}
			}
			((float *)entropy.data)[current_index_in_origin] = e;
		}
	}
	free(neighbood_offset);
	free(image_cumprod);
	free(cumprod);
	free(neighbood_corrds_array);

	func_end = clock();
	double func_time = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	std::cout << "func time" << func_time << std::endl;
}

//attempt at entropy based lesion detection/localization.
//Based from process in a paper, "Acne image analysis: lesion localization and classification"
std::vector<std::vector<cv::Point>> lesion_detect_entropy(const cv::Mat & image, std::string img_name) {
	///VARIABLES / SETTINGS
	//all tuning/performance parameters in one place.
	int gauss_ksize = 11;
	int size_open = 1;
	int size_close = 1;
	int size_open2 = 3;
	int size_close2 = 0;

	//guassian blur
	cv::Size ksize;
	ksize.height = gauss_ksize;
	ksize.width = ksize.height;
	//thresholding
	int thresh_val = 0;
	int maxValue = 255;
	int thresholdType = cv::THRESH_BINARY_INV;
	int adaptMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	//morphology
	int shape = cv::MORPH_ELLIPSE;
	int open = cv::MORPH_OPEN;
	int close = cv::MORPH_CLOSE;
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
	//entropy
	cv::Rect roi(0, 0, image.cols, image.rows);
	cv::Mat dst = cv::Mat::zeros(image.rows, image.cols, CV_32F);
	cv::Mat dst_show(image.rows, image.cols, CV_32FC1);
	//countour
	std::vector<std::vector<cv::Point>> contours;
	//matrices for each step of process
	cv::Mat gr_img, blur_img, gray_img, close_img, open_img, dilate_img;
	cv::Mat1b thresh_img, thresh1, thresh2, thresh3, thresh4, entropy;

	///PROCESS
	/*
	1: removing blue channel removes noise (green-red image)
	2: guassin blur filter to reduce image noise
	3: convert to gray scale
	4: local entropy
	5: threshold
	6: close to fill in gaps
	7: open removes smaller blobs
	8: dilate what's left to make them more prominent
	9: find contours, the points that make up border of area on the original image
	*/

	gr_img = image & cv::Scalar(0, 255, 255);
	cv::GaussianBlur(gr_img, blur_img, ksize, 0);
	cv::cvtColor(blur_img, gray_img, CV_BGR2GRAY);
	//entropy
	getLocalEntropyImage(gray_img, roi, dst);
	cv::normalize(dst, dst_show, 0.0, 255.0, cv::NORM_MINMAX, CV_32FC1);
	dst_show.convertTo(entropy, CV_8U);
	
	///Adaptive thresholding was mentioned in the paper, however I found that it didn't really work.
	///In same paper they also mention 'multi-thresholding' and 'qunatifying the image'. 
	///Judging from the sample images and googling that term, it appears that they used the Matlab 
	/// function "multithresh" with one level, so it's essentially an Otsu's threshold.
	///Below I just used a hard-coded threshold to experiment with. Plan is Otsu's threshold
	
	///adpative thresholding (do not advise)
	//int blocksize = 125;
	//cv::adaptiveThreshold(entropy, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, -15);  
	//cv::bitwise_not(thresh_img, thresh_img);
	
	///plain-old Otsu's threshold
	//int thresh_val = cv::threshold(entropy, thresh_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	
	
	///2 levels of Otsu's threshold
	cv::threshold(entropy, thresh1, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	thresh_val = threshold_with_mask(entropy, thresh2, 0, 255, CV_THRESH_OTSU, thresh1);
	cv::bitwise_and(thresh2, thresh1, thresh3);
	threshold_with_mask(entropy, thresh4, thresh_val, 255, CV_THRESH_BINARY, thresh3);
	cv::bitwise_and(thresh3, thresh4, thresh_img);

	cv::morphologyEx(thresh_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open2);
	cv::morphologyEx(open_img, close_img, close, elem_close2);

	cv::findContours(close_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));


	///------------- TESTING / DEBUG ---------------------
	std::string img_out_dir = output_dir + "/detection_entropy/";
	_mkdir(img_out_dir.c_str());
	img_out_dir = img_out_dir + img_name + "/";
	_mkdir(img_out_dir.c_str());

	FILE * file;
	std::string close_file = img_out_dir + "/num_of_lesions.txt";
	file = fopen(close_file.c_str(), "w");

	cv::bitwise_not(close_img, dilate_img);
	cv::Mat masked, color;
	cv::cvtColor(dilate_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked);

	//cv::imwrite(img_out_dir + "_thresh1_" + img_name + ".jpg", thresh1);
	//cv::imwrite(img_out_dir + "_thresh2_" + img_name + ".jpg", thresh2);
	//cv::imwrite(img_out_dir + "_thresh3_" + img_name + ".jpg", thresh3);
	//cv::imwrite(img_out_dir + "_thresh4_" + img_name + ".jpg", thresh4);
	//cv::imwrite(img_out_dir + "_thresh5_" + img_name + ".jpg", thresh5);
	//cv::imwrite(img_out_dir + "_thresh3_" + img_name + ".jpg", thresh6);
	
	cv::imwrite(img_out_dir + "_0_bgr.jpg", image);
	//cv::imwrite(img_out_dir + "_1_gr_" + img_name + ".jpg", gr_img);
	//cv::imwrite(img_out_dir + "_2_blur_" + img_name + ".jpg", blur_img);
	cv::imwrite(img_out_dir + "_3_entropy_" + img_name + ".jpg", entropy);
	cv::imwrite(img_out_dir + "_4_thresh_" + img_name + ".jpg", thresh_img);
	cv::imwrite(img_out_dir + "_5_close_" + img_name + ".jpg", close_img);
	cv::imwrite(img_out_dir + "_6_open_" + img_name + ".jpg", open_img);
	//cv::imwrite(img_out_dir + "_7_dilate_" + img_name + ".jpg", dilate_img);
	cv::imwrite(img_out_dir + "_8_masked_" + img_name + ".jpg", masked);
	
	int num_lesions = contours.size() - 1;
	fprintf(file, "# lesions: %d \n", num_lesions);
	///--------------------------------------------------------

	return contours;
	
}
