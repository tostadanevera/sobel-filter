#include "sobel_v1.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

#include <iostream>

#include <chrono>

namespace sobel_v1 {

// Sobel kernel 3x3, detects vertical edges
constexpr int kernelX[3][3] = {
    {-1, 0, 1},    
    {-2, 0, 2},
    {-1, 0, 1},
};

// Sobel kernel 3x3, detects horizontal edges
constexpr int kernelY[3][3] = {
    {-1, -2, -1},    
    { 0,  0,  0},
    { 1,  2,  1},
};

cv::Mat apply_sobel(const cv::Mat& input) 
{
    CV_Assert(input.type() == CV_64FC1 && "apply_sobel expects CV_64FC1");
    CV_Assert(!input.empty() && "apply_sobel received an empty image");

    const int rows = input.rows;
    const int cols = input.cols;

    cv::Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);

    #pragma omp parallel for default(none) shared(input, output, rows, cols, kernelX, kernelY) schedule(static)
    for(int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double gradientX = 0.0, gradientY = 0.0;

            for (int kernelRow = -1; kernelRow <= 1; ++kernelRow) {
                const int sampleRow = std::clamp(row + kernelRow, 0, rows - 1);
                const double* inputRowPtr = input.ptr<double>(sampleRow);

                for (int kernelCol = -1; kernelCol <= 1; ++kernelCol) {
                    const int sampleCol = std::clamp(col + kernelCol, 0, cols - 1);

                    gradientX += kernelX[kernelRow + 1][kernelCol+1] * inputRowPtr[sampleCol];
                    gradientY += kernelY[kernelRow + 1][kernelCol+1] * inputRowPtr[sampleCol];
                }
            }

            const double gradient = std::sqrt(gradientX * gradientX + gradientY * gradientY);

            output.at<uchar>(row, col) = 
                cv::saturate_cast<uchar>(gradient);
        }
    }

    return output;
}

} // namespace