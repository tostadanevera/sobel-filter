#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
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

    // Interior pixels only
    for(int i = 1; i < rows - 1; i++) {

        const double* Mt = M.ptr<double>(i-1); // top
        const double* Mi = M.ptr<double>(i);
        const double* Mb = M.ptr<double>(i+1); // bottom
        for (int j = 1; j < cols - 1; j++) {
            int gx = 0, gy = 0;
            // std::cout << Mi[j] << " ";

            gx += KX[0][0] * Mt[j-1];
            gx += KX[0][1] * Mt[j];
            gx += KX[0][2] * Mt[j+1];  
            gx += KX[1][0] * Mi[j-1];
            gx += KX[1][1] * Mi[j];
            gx += KX[1][2] * Mi[j+1];  
            gx += KX[2][0] * Mb[j-1];
            gx += KX[2][1] * Mb[j];
            gx += KX[2][2] * Mb[j+1];
            
            gy += KY[0][0] * Mt[j-1];
            gy += KY[0][1] * Mt[j];
            gy += KY[0][2] * Mt[j+1];  
            gy += KY[1][0] * Mi[j-1];
            gy += KY[1][1] * Mi[j];
            gy += KY[1][2] * Mi[j+1];  
            gy += KY[2][0] * Mb[j-1];
            gy += KY[2][1] * Mb[j];
            gy += KY[2][2] * Mb[j+1];   

            // for (int k = 0; k < 3; k++) {
            //     for (int l = 0; l < 3; l++) {
            //         gx += ;
            //         gy += ;
            //     }
            // }

            int gradient = std::sqrt(static_cast<float>(gx * gx + gy * gy));

            std::cout << gradient << " ";

            output.at<uchar>(i, j) = static_cast<uchar>(gradient);
        }

        std::cout << std::endl;
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