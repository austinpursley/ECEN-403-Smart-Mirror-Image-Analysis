//note: put after finding contour lesions

///COLOR SPACE EXPERIMENTS
	cv::Mat3b lab_img(image.rows, image.cols, CV_8UC3);
	cv::Mat3b yrb_img(image.rows, image.cols, CV_8UC3);
	cv::Mat3b hsv_img(image.rows, image.cols, CV_8UC3);
	cv::Mat ab_img, cr_img;
	cv::cvtColor(gr_img, lab_img, CV_BGR2Lab, 3);
	cv::cvtColor(gr_img, yrb_img, CV_BGR2YCrCb, 3);
	cv::cvtColor(blur_img, hsv_img, CV_BGR2HSV, 3);
	std::vector<cv::Mat1b> bgr(3);
	std::vector<cv::Mat1b> lab(3);
	std::vector<cv::Mat1b> yrb(3);
	std::vector<cv::Mat1b> hsv(3);

	cv::split(gr_img, bgr);
	cv::split(lab_img, lab);
	cv::split(yrb_img, yrb);
	cv::split(hsv_img, hsv);
	
	cv::Mat red, dark, all;
	cv::Mat AB, AnotB;

	std::vector<std::vector<cv::Point>> contours1;
	std::vector<std::vector<cv::Point>> contours2;
	std::vector<std::vector<cv::Point>> contours3;
	cv::Mat morph1;
	cv::Mat morph2;
	cv::Mat morph3;


	cv::GaussianBlur(image, blur_img, ksize, 0);
	cv::cvtColor(blur_img, gray_img, CV_BGR2GRAY);
	std::vector<std::vector<cv::Point>> contours1_filt;
	std::vector<std::vector<cv::Point>> contours2_filt;
	std::vector<std::vector<cv::Point>> contours3_filt;

	//------- LIGHT RED -----------//
	
	cv::Scalar mean_color = cv::mean(image);
	cv::Mat mean_gray_img, redd;
	cv::Mat mean_color_img(image.rows, image.cols, CV_8UC3, mean_color);
	cv::cvtColor(mean_color_img, mean_gray_img, CV_BGR2GRAY, 1);
	cv::bitwise_not(yrb[1], yrb[1]);
	cv::addWeighted(yrb[1], 0.6, mean_gray_img, 0.4, 0, redd);
	/*
	cv::adaptiveThreshold(redd, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close2);
	cv::morphologyEx(close_img, morph1, open, elem_open2);
	cv::findContours(morph1, contours1, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	filter_lesions_by_entropy(gray_img, contours1, contours1_filt, (img_name + "_0"), 5, 0.3);
	*/

	//-------- DARK / MOLES --------//
	//cv::bitwise_not(lab[2], lab[2]);
	cv::bitwise_not(yrb[1], yrb[1]);
	//cv::addWeighted(yrb[1], 0.3, lab[2], 0.7, 0, dark);
	cv::addWeighted(yrb[1], 0.3, lab[2], 0.7, 0, AnotB);
	cv::addWeighted(AnotB, 0.9, lab[0], 0.1, 0, dark);

	cv::adaptiveThreshold(dark, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close2);
	cv::morphologyEx(close_img, morph2, open, elem_open2);
	cv::morphologyEx(morph2, morph2, erode, elem_erode);
	cv::findContours(morph2, contours2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//filter_lesions_by_entropy(gray_img, contours2, contours2_filt, (img_name + "_0"), 5, 0.3);
	color_filter(image, contours2, (img_name + "_0"), 5, 0.15);
	//color_filter(image, contours2, (img_name + "_0"), 5, 0.5);

	cv::bitwise_not(lab[2], lab[2]);
	//---------- ALL ------------//
	/*
	cv::addWeighted(lab[1], 0.5, lab[2], 0.5, 0, AB);
	cv::addWeighted(AB, 0.6, lab[0], 0.4, 0, all);

	cv::adaptiveThreshold(all, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close2);
	cv::morphologyEx(close_img, morph3, open, elem_open2);
	cv::findContours(morph3, contours3, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	filter_lesions_by_entropy(gray_img, contours3, contours3_filt, (img_name + "_1"), 5, 0.3);
	*/

	//color filtering
	//std::vector<std::vector<cv::Point>> filtered_cnts;
	//color_filter(blur_img, contours, filtered_cnts, (img_name + "_0"), 5, 0.25);

	///---------------------
	cv::Mat masked1_filter, masked2_filter, masked3_filter, masked3_combined;
	cv::Mat masked1, masked2, masked3, color;
	image.copyTo(masked1_filter);
	image.copyTo(masked2_filter);
	image.copyTo(masked3_filter);

	image.copyTo(masked3_combined);
	//cv::drawContours(masked1_filter, contours1_filt, -1, cv::Scalar(255), -1);
	cv::drawContours(masked2_filter, contours2, -1, cv::Scalar(255), -1);
	//cv::drawContours(masked3_filter, contours3_filt, -1, cv::Scalar(255), -1);

	//cv::drawContours(masked3_combined, contours1_filt, -1, cv::Scalar(255), -1);
	//cv::drawContours(masked3_combined, contours3_filt, -1, cv::Scalar(255), -1);
	
	//cv::bitwise_not(morph1, morph1);
	//cv::cvtColor(morph1, color, CV_GRAY2BGR);
	//cv::bitwise_and(color, image, masked1);

	cv::bitwise_not(morph2, morph2);
	cv::cvtColor(morph2, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked2);

	//cv::bitwise_not(morph3, morph3);
	//cv::cvtColor(morph3, color, CV_GRAY2BGR);
	//cv::bitwise_and(color, image, masked3);
	
		cv::imwrite(img_out_dir + img_name + "_0_bgr.jpg", image);
	cv::imwrite(img_out_dir + img_name + "_1_masked2.jpg", masked2);
	cv::imwrite(img_out_dir + img_name + "_5_masked2_filter_" + ".jpg", masked2_filter);