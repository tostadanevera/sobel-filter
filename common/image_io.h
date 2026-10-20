#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace image_io {
    // Loads `path` as an 8-bit grayscale image and converts it to CV_64FC1
    // Returns an empty cv::Mat if the file cannot be read
    cv::Mat load_grayscale(const std::string& path);
}