#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace sobel_v3 {
    cv::Mat apply_sobel(const cv::Mat& input);
    cv::Mat load_image(const std::string& imagePath);
}