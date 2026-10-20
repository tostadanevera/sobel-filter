#include "./common/image_io.h"
#include "./v1/sobel_v1.h"
#include "./v2/sobel_v2.h"
#include "./v3/sobel_v3.h"

#include <opencv2/opencv.hpp>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {

    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image> [iterations]\n";
        return 1;
    }

    const cv::Mat img = image_io::load_grayscale(argv[1]);
    if (img.empty()) {
        std::cerr << "Image could not be read\n";
        return 1;
    }

    int iterations = 10;
    if (argc == 3) {
        try {
            iterations = std::stoi(argv[2]);
        } catch (const std::exception&) {
            std::cerr << "Invalid iteration count: " << argv[2] << '\n';
            return 1;
        }
        if (iterations <= 0) {
            std::cerr << "Iterations must be > 0\n";
            return 1;
        }
    }

    auto bench = [&](auto fn) {
        fn(img);  // warmup
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            cv::Mat res = fn(img);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
    };

    const int maxThreads = omp_get_max_threads();

    std::printf("image: %dx%d, iterations: %d, max threads: %d\n\n", img.cols, img.rows, iterations, maxThreads);
    std::printf("%7s %10s %10s %10s\n", "threads", "sobel_v1", "sobel_v2", "sobel_v3");
    std::printf("%7s %10s %10s %10s\n", "-------", "--------", "--------", "--------");

    for (int t = 1; t <= maxThreads; t *= 2) {
        omp_set_num_threads(t);
        const double v1 = bench(&sobel_v1::apply_sobel);
        const double v2 = bench(&sobel_v2::apply_sobel);
        const double v3 = bench(&sobel_v3::apply_sobel);
        std::printf("%7d %10.3f %10.3f %10.3f\n", t, v1, v2, v3);
    }

    omp_set_num_threads(maxThreads);  // restore
    return 0;
}