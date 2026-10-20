#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace sobel_v2 {
    cv::Mat apply_sobel(const cv::Mat& input);
}