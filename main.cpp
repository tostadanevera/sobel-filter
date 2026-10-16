#include <opencv2/opencv.hpp>

#include <cmath>
#include <algorithm>

#include <iostream>

using namespace cv;

const int KX[3][3] = {{-1, 0, 1},    
                      {-2, 0, 2},
                      {-1, 0, 1}};

const int KY[3][3] = {{-1, -2, -1},    
                      {0, 0, 0},
                      {1, 2, 1}};

Mat sobel_filter(Mat& M) 
{
    int cols = M.cols;
    int rows = M.rows;

    Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);

    // TODO: Fix outer pixels
    // Interior pixels only
    for(int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            double gx = 0, gy = 0;

            for (int k = -1; k < 2; k++) {
                const double* Mc = M.ptr<double>(i+k);
                for (int l = -1; l < 2; l++) {
                    gx += KX[k+1][l+1] * Mc[j+l];
                    gy += KY[k+1][l+1] * Mc[j+l];
                }
            }

            double gradient = std::sqrt(static_cast<float>(gx * gx + gy * gy));

            // std::cout << gradient << " ";

            output.at<uchar>(i, j) = static_cast<uchar>(gradient);
        }
        // std::cout << std::endl;
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

    // Show image

    // cv::namedWindow("img");

    // imshow("Display window", img);
    // int k = waitKey(0); // Wait for a keystroke in the window

    // if(k == 's')
    // {
    //     imwrite("./output/kitteh.png", img);
    // }
    
    
}