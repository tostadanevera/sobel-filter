#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace sobel_v3 {

    // Separable Sobel filter, the 1D derivatives are computed once and 
    // reused for both Gx and Gy, then combined with the smoothing weights 
    // Fastest version, output matches v1/v2
    cv::Mat apply_sobel(const cv::Mat& input);
}