#include <opencv2/opencv.hpp>
#include <algorithm>

#include <iostream>

using namespace cv;

const int KX[3][3] = {{-1, 0, 1},    
                      {-2, 0, 2},
                      {-1, 0, 1}};

const int KY[3][3] = {{-1, -2, -1},    
                      {0, 0, 0},
                      {1, 2, 1}};

Mat sobel_filter(const Mat& M) 
{
    int cols = M.cols;
    int rows = M.rows;

    Mat output = Mat::zeros(rows, cols, CV_8UC1);

    #pragma omp parallel for
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double gx = 0, gy = 0;
            for (int k = -1; k < 2; k++) {
                int row = std::clamp(i + k, 0, rows - 1);
                const double* Mr = M.ptr<double>(row);
                for (int l = -1; l < 2; l++) {
                    int col = std::clamp(j + l, 0, cols - 1);
                    gx += KX[k+1][l+1] * Mr[col];
                    gy += KY[k+1][l+1] * Mr[col];
                }
            }

            double gradient = std::sqrt(gx * gx + gy * gy);

            output.at<uchar>(i, j) = saturate_cast<uchar>(gradient);
        }
    }

    return output;
}

int main()
{
    // Read image
    std::string image_path = "./media/kitteh.jpeg";
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty())
    {
        std::cout << "Image empty" << std::endl;
        return -1;
    }

    // Convert to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat gray64;
    gray.convertTo(gray64, CV_64F);
 
    // Apply filter
    Mat res = sobel_filter(gray64);

    // Write image
    imwrite("res.png", res);    
}