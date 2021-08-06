#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>

#include "Normalization.h"
#include "Segmentation.h"


int main() {


	cv::Mat image = cv::imread("img_2.png");
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

	cv::imshow("Normalized image",normalized);
	cv::waitKey(0);
	cv::destroyWindow("Normalized image");
	int block_size = 16;
	returnData data = create_segmented_and_variance_images(normalized, block_size, 0.2);
	cv::Mat segmented_image = data.segmented_image;
	cv::Mat norm_img = data.norm_img;
	cv::Mat mask = data.mask;

	cv::imshow("Segmented image", segmented_image);
	cv::waitKey(0);
	cv::destroyWindow("Segmented image");

	cv::imshow("Norm image", norm_img);
	cv::waitKey(0);
	cv::destroyWindow("Norm image");

	cv::imshow("Mask", mask);
	cv::waitKey(0);
	cv::destroyWindow("Mask");
	
	
	

	
	

	

	return 0;
}