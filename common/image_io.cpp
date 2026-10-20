#include "image_io.h"

namespace image_io {

cv::Mat load_grayscale(const std::string& path)
{
    const cv::Mat color = cv::imread(path, cv::IMREAD_COLOR);
    if (color.empty()) {
        return {};
    }

    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    cv::Mat gray64f;
    gray.convertTo(gray64f, CV_64F);
    return gray64f;
}

} // namespace