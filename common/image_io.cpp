#include "image_io.h"

namespace image_io {

cv::Mat load_grayscale(const std::string& path)
{
    const cv::Mat color = cv::imread(path, cv::IMREAD_COLOR);
    if (color.empty()) {
        return {};
    }

    // Transform to grayscale
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    // 8-bit to double, to avoid overflow
    cv::Mat gray64f;
    gray.convertTo(gray64f, CV_64F);
    return gray64f;
}

} // namespace