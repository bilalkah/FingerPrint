#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>

#include "Normalization.h"


int main() {


	cv::Mat image = cv::imread("lotus.jpg");
	if (image.empty()) {
		return -1;
	}

	cv::Mat gray_image;
	cvtColor(image, gray_image,cv::COLOR_BGR2GRAY);
	
	double meanImg = meanOfImg(gray_image);
	std::cout << "Mean of image:" << meanImg << std::endl;
	double stdImg = stdOfImg(gray_image);
	std::cout << "Standard deviation of image: " << stdImg << std::endl;
	cv::Mat normalized = normalize(gray_image, 100, 100);
	imshow("Normalized", normalized);
	cv::waitKey(0);
	cv::destroyWindow("Normalized");


	return 0;
}