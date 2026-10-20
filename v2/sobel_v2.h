#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace sobel_v2 {

    // Unrolled Sobel filter, same math as v1, but the 3x3 convolution is
    // written out explicitly, and pixel access uses cached row pointers
    // Output matches v1
    cv::Mat apply_sobel(const cv::Mat& input);
}