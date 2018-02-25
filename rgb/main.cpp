// RGB_img_full_process.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "face_feature_extract/face_feature_extract.h"
#include "dirent.h"

int main()
{
	std::string dir = "C:/Users/Austin Pursley/Desktop/ECEN-Senior-Design-Smart-Mirror-Image-Processing/rgb/";
	std::string input_dir = dir + "input/";
	std::string output_dir = dir + "output/";

	//get list of images in input direcotry
	DIR *dpdf;
	struct dirent *epdf;
	std::vector<std::string> filenames;
	dpdf = opendir(input_dir.c_str());
	if (dpdf != NULL) {
		while (epdf = readdir(dpdf)) {
			filenames.push_back(std::string(epdf->d_name));
		}
	}

	//for each image in input directory
	for (int i = 2; i < filenames.size(); i++) {
		//read image
		std::string img_file = filenames[i]; //e.g. name.jpg
		size_t lastindex = img_file.find_last_of(".");
		std::string img_name = img_file.substr(0, lastindex); //e.g name (no .jpg extension)
		std::string img_dir(input_dir.c_str() + img_file);
		cv::Mat img = cv::imread(img_dir, cv::IMREAD_COLOR);
		if (!img.data) {
			std::cout << "Unable to open the file: " << img_dir;
			return 1;
		}
		
		///STEP 1: detect the face, extract the eyes and mouth
		std::string face_cascade_name = "C:/Users/Austin Pursley/Documents/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
		std::string eyes_cascade_name = "C:/Users/Austin Pursley/Documents/opencv/build/etc/haarcascades/haarcascade_eye.xml";
		std::string mouth_cascade_name = "C:/Users/Austin Pursley/Documents/opencv/build/etc/haarcascades/Mouth.xml";
		cv::CascadeClassifier face_cascade;
		cv::CascadeClassifier eyes_cascade;
		cv::CascadeClassifier mouth_cascade;
		// Load the face cascade
		if (!face_cascade.load(face_cascade_name)) {
			printf("--(!)Error loading face cascade\n");
			return (-1);
		}
		// Load the eyes cascade
		if (!eyes_cascade.load(eyes_cascade_name)) {
			printf("--(!)Error loading eyes cascade\n");
			return (-1);
		}
		// Load the mouth cascade
		if (!mouth_cascade.load(mouth_cascade_name)) {
			printf("--(!)Error loading mouth cascade\n");
			return (-1);
		}
		// Apply the classifier to the frame
		std::map<std::string, cv::Rect> face_featrs;
		if (!img.empty()) {
			get_face_features(img, face_cascade, eyes_cascade, mouth_cascade, face_featrs);
		}
		else {
			printf(" --(!) No captured frame -- Break!");
			exit(1);
		}

		cv::Mat face = img(face_featrs["face"]);
		cv::Mat face_masked;
		mask_features(face, face_featrs, face_masked);

		cv::imwrite(output_dir + img_name + "_0_img.jpg", img);
		cv::imwrite(output_dir + img_name + "_1_face.jpg", face);
		cv::imwrite(output_dir + img_name + "_2_face_features_masked.jpg", face_masked);

		///STEP 1: SKIN EXTRACTION
		cv::Mat lab_image;
		//cvtColor(matImage, lab_image, cv::COLOR_BGR2Lab);
		cvtColor(img, lab_image, cv::COLOR_RGB2HSV);
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

		lower[2] = lower[2] -40;
		upper[2] = upper[2] + 100;

		//in range
		cv::Mat range;
		cv::inRange(lab_image, lower, upper, range);

		cvtColor(crop, crop, cv::COLOR_HSV2RGB);
		cv::Mat crop_mask;
		//crop.copyTo(crop_mask, thresh);
		crop.copyTo(crop_mask);
		cv::imwrite(output_dir + img_name + "_3_crop.jpg", crop_mask);
		cv::imwrite(output_dir + img_name + "_4_range.jpg", range);



		///NEXT/ISSUES: normalizing face, question of if should have "original" image and then have scaled image of face (face detection) that's used for all image processing.
		///probably have to add new output to face feature function (or break that function up?)

	}
    return 0;
}

