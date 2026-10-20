#include "sobel_v3.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

#include <iostream>

#include <chrono>

namespace sobel_v3 {

// No 2D kernels, the Sobel kernels are separable
// Instead applies the 1D derivative [-1, 0, 1] and the 1D smoothing [1, 2, 1]

cv::Mat apply_sobel(const cv::Mat& input) 
{
    CV_Assert(input.type() == CV_64FC1 && "apply_sobel expects CV_64FC1");
    CV_Assert(!input.empty() && "apply_sobel received an empty image");

    const int rows = input.rows;
    const int cols = input.cols;

    cv::Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);

    // Parallelize over rows
    #pragma omp parallel for default(none) shared(input, output, rows, cols) schedule(static)
    for (int row = 0; row < rows; ++row) {
        
        // Border pixels, replicate the nearest valid row
        const int rowM1 = std::clamp(row - 1, 0, rows - 1);
        const int rowP1 = std::clamp(row + 1, 0, rows - 1);

        const double* ptrM1 = input.ptr<double>(rowM1);
        const double* ptr0  = input.ptr<double>(row);
        const double* ptrP1 = input.ptr<double>(rowP1);
        
        uchar* outRow = output.ptr<uchar>(row);

        for (int col = 0; col < cols; ++col) {
            // Border pixels, replicate the nearest valid column
            const int colM1 = std::clamp(col - 1, 0, cols - 1);
            const int colP1 = std::clamp(col + 1, 0, cols - 1);

            // Horizontal derivative [-1, 0, 1]
            const double dxTop = ptrM1[colP1] - ptrM1[colM1];
            const double dxMid = ptr0[colP1]  - ptr0[colM1];
            const double dxBot = ptrP1[colP1] - ptrP1[colM1];

            // Vertical derivative [-1, 0, 1]
            const double dyLeft  = ptrP1[colM1] - ptrM1[colM1];
            const double dyMid   = ptrP1[col]   - ptrM1[col];
            const double dyRight = ptrP1[colP1] - ptrM1[colP1];

            // Smoothing [1, 2, 1]
            // Gx, smooth dx vertically
            const double gradientX = dxTop * 1.0 + dxMid * 2.0 + dxBot * 1.0;

            // Gy, smooth dy horizontally
            const double gradientY = dyLeft * 1.0 + dyMid * 2.0 + dyRight * 1.0;

            const double gradient = std::sqrt(gradientX * gradientX + gradientY * gradientY);
            outRow[col] = cv::saturate_cast<uchar>(gradient); // Rounds, magnitude can exceed 255
        }
    }

    return output;
}

} // namespace