#include "./common/image_io.h"
#include "./v3/sobel_v3.h"
#include <iostream>

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_input> <image_output>\n";
        return EXIT_FAILURE;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];

    const cv::Mat img = image_io::load_grayscale(inputPath);
    
    if (img.empty()) {
        std::cerr << "Image could not be read: " << inputPath << '\n';
        return EXIT_FAILURE;
    }

    const cv::Mat res = sobel_v3::apply_sobel(img);

    if (!cv::imwrite(outputPath, res)) {
        std::cerr << "Image could not be saved: " << outputPath << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Image saved at: " << outputPath << '\n';

    return EXIT_SUCCESS;
}