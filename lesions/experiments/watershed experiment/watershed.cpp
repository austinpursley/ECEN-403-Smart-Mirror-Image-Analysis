///watershed experiment
//note: put this after finding contours of lesions

	cv::Mat dist;
	cv::distanceTransform(morph2, dist, CV_DIST_L2, 3);
	normalize(dist, dist, 0, 255., cv::NORM_MINMAX);
	dist.convertTo(dist, CV_8UC1);
	cv::Mat sure_foreground;
	cv::threshold(dist, sure_foreground, 100, 255, CV_THRESH_BINARY);
	
	cv::Mat sure_background;
	cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
	int iterations = 3;
	cv::dilate(morph2, sure_background, kernel, cv::Point(-1,-1), 3);

	cv::Mat unknown;
	cv::subtract(sure_background, sure_foreground, unknown);
	cv::erode(unknown, unknown, kernel, cv::Point(-1, -1), 1);

	std::vector<std::vector<cv::Point> > marker_contours;
	findContours(sure_foreground, marker_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	std::vector<std::vector<cv::Point> > unknown_contours;
	findContours(unknown, unknown_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
	markers = markers + 1;
	//cv::connectedComponents(sure_foreground, markers);
	//markers = markers + 1;

	for (size_t i = 0; i < unknown_contours.size(); i++)
		drawContours(markers, unknown_contours, i, 0, -1);

	for (size_t i = 0; i < marker_contours.size(); i++)
		drawContours(markers, marker_contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i) + 2), -1);
	
	

	cv::Mat marker_show;
	markers.convertTo(marker_show, CV_8UC3);
	normalize(marker_show, marker_show, 0, 255., cv::NORM_MINMAX);
	
	watershed(image, markers);
	
	//markers.convertTo(markers, CV_8UC1);
	std::vector<std::vector<cv::Point> > watershed_contours;
	cv::findContours(markers, watershed_contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//cv::Mat watershed;
	//image.copyTo(watershed);
	//cv::drawContours(watershed, watershed_contours, -1, cv::Scalar(255), -1);
	
	// Create the result image
	std::vector<cv::Vec3b> colors;
	for (size_t i = 0; i < marker_contours.size(); i++)
	{
		int b = cv::theRNG().uniform(0, 255);
		int g = cv::theRNG().uniform(0, 255);
		int r = cv::theRNG().uniform(0, 255);
		colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(marker_contours.size()))
				dst.at<cv::Vec3b>(i, j) = colors[index - 1];
			else
				dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
		}
	}
	
	/*
	cv::imwrite(img_out_dir + img_name + "_2_fg_" + ".jpg", sure_foreground);
	cv::imwrite(img_out_dir + img_name + "_3_bg_" + ".jpg", sure_background);
	cv::imwrite(img_out_dir + img_name + "_4_unknown_" + ".jpg", unknown);
	cv::imwrite(img_out_dir + img_name + "_5_markers_" + ".jpg", marker_show);
	cv::imwrite(img_out_dir + img_name + "_6_watershed_" + ".jpg", dst);
	*/

	///---------------------------