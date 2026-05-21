#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>

#include "stereo_batch_preprocess.h"

namespace
{

void require(bool value, const std::string& message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

void test_scale_one_preserves_image()
{
    cv::Mat image(8, 12, CV_8UC3, cv::Scalar(10, 20, 30));

    cv::Mat scaled = stereo_examples::scale_image_for_sgm(image, 1.0);

    require(scaled.size() == image.size(), "scale 1.0 should preserve image size");
    require(scaled.type() == image.type(), "scale 1.0 should preserve image type");
}

void test_scale_down_uses_requested_ratio()
{
    cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(10, 20, 30));

    cv::Mat scaled = stereo_examples::scale_image_for_sgm(image, 0.5);

    require(scaled.size() == cv::Size(960, 540), "scale 0.5 should halve a 1920x1080 image");
    require(scaled.type() == image.type(), "scaled image type changed");
}

void test_rejects_invalid_scale()
{
    bool threw_zero = false;
    bool threw_large = false;

    try
    {
        stereo_examples::validate_sgm_scale(0.0);
    }
    catch (const std::runtime_error&)
    {
        threw_zero = true;
    }

    try
    {
        stereo_examples::validate_sgm_scale(1.1);
    }
    catch (const std::runtime_error&)
    {
        threw_large = true;
    }

    require(threw_zero, "scale 0.0 should be rejected");
    require(threw_large, "scale greater than 1.0 should be rejected");
}

} // namespace

int main()
{
    try
    {
        test_scale_one_preserves_image();
        test_scale_down_uses_requested_ratio();
        test_rejects_invalid_scale();
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_stereo_batch_preprocess failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
