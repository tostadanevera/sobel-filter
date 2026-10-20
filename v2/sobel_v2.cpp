#include "sobel_v2.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

#include <iostream>

#include <chrono>

namespace sobel_v2 {

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

    // Parallelize over rows
    #pragma omp parallel for default(none) shared(input, output, rows, cols, kernelX, kernelY) schedule(static)
    for (int row = 0; row < rows; ++row) {
        
        // Input row pointers
        const int rowM1 = std::clamp(row - 1, 0, rows - 1); // replicate the nearest valid column in case of border pixel
        const int row0  = row;
        const int rowP1 = std::clamp(row + 1, 0, rows - 1); // replicate the nearest valid column in case of border pixel

        const double* ptrM1 = input.ptr<double>(rowM1);
        const double* ptr0  = input.ptr<double>(row0);
        const double* ptrP1 = input.ptr<double>(rowP1);
        
        // Output row pointer
        uchar* outRow = output.ptr<uchar>(row);

        for (int col = 0; col < cols; ++col) {
            double gradientX = 0.0, gradientY = 0.0;

            // Column indices
            const int colM1 = std::clamp(col - 1, 0, cols - 1);
            const int col0  = col;
            const int colP1 = std::clamp(col + 1, 0, cols - 1);

            // Row -1
            gradientX += kernelX[0][0] * ptrM1[colM1] + kernelX[0][1] * ptrM1[col0] + kernelX[0][2] * ptrM1[colP1];
            gradientY += kernelY[0][0] * ptrM1[colM1] + kernelY[0][1] * ptrM1[col0] + kernelY[0][2] * ptrM1[colP1];

            // Row 0
            gradientX += kernelX[1][0] * ptr0[colM1] + kernelX[1][1] * ptr0[col0] + kernelX[1][2] * ptr0[colP1];
            gradientY += kernelY[1][0] * ptr0[colM1] + kernelY[1][1] * ptr0[col0] + kernelY[1][2] * ptr0[colP1];

            // Row +1
            gradientX += kernelX[2][0] * ptrP1[colM1] + kernelX[2][1] * ptrP1[col0] + kernelX[2][2] * ptrP1[colP1];
            gradientY += kernelY[2][0] * ptrP1[colM1] + kernelY[2][1] * ptrP1[col0] + kernelY[2][2] * ptrP1[colP1];

            const double gradient = std::sqrt(gradientX * gradientX + gradientY * gradientY);
            outRow[col] = cv::saturate_cast<uchar>(gradient); // Rounds, magnitude can exceed 255
        }
    }

    return output;
}

} // namespace