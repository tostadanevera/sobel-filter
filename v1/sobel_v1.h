#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace sobel_v1 {
    
    // Implementation of the Sobel filter, slowest of the three versions
    // Convolves a CV_64FC1 grayscale image with the two 3x3 Sobel kernels
    // using nested loops, then writes sqrt(Gx^2 + Gy^2) clamped to [0, 255]
    // into a CV_8UC1 image of the same size
    cv::Mat apply_sobel(const cv::Mat& input);
}