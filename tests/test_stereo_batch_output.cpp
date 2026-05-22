#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>

#include "stereo_batch_output.h"

namespace
{

void require(bool value, const std::string& message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

bool is_black(const cv::Vec3b& value)
{
    return value[0] == 0 && value[1] == 0 && value[2] == 0;
}

void test_converts_8bit_disparity_to_color()
{
    cv::Mat disparity(2, 3, CV_8UC1, cv::Scalar(64));
    disparity.at<unsigned char>(0, 0) = 255;
    cv::Mat output = stereo_examples::prepare_disparity_for_png(disparity, 128, false, 255);

    require(output.type() == CV_8UC3, "8-bit disparity should be saved as RGB PNG data");
    require(output.size() == disparity.size(), "8-bit color disparity size changed");
    require(is_black(output.at<cv::Vec3b>(0, 0)), "invalid 8-bit disparity should be black");
    require(!is_black(output.at<cv::Vec3b>(0, 1)), "valid 8-bit disparity should be colorized");
}

void test_converts_16bit_subpixel_disparity_to_color()
{
    cv::Mat disparity(2, 3, CV_16UC1, cv::Scalar(1024));
    disparity.at<unsigned short>(0, 0) = 65520;
    cv::Mat output = stereo_examples::prepare_disparity_for_png(disparity, 128, true, 65520);

    require(output.type() == CV_8UC3, "16-bit disparity should be saved as RGB PNG data");
    require(output.size() == disparity.size(), "16-bit color disparity size changed");
    require(is_black(output.at<cv::Vec3b>(0, 0)), "invalid 16-bit disparity should be black");
    require(!is_black(output.at<cv::Vec3b>(0, 1)), "valid 16-bit disparity should be colorized");
}

void test_converts_8bit_raw_disparity_to_16bit_png_data()
{
    cv::Mat disparity(2, 3, CV_8UC1, cv::Scalar(64));
    disparity.at<unsigned char>(0, 0) = 255;

    cv::Mat output = stereo_examples::prepare_raw_disparity_for_png(disparity);

    require(output.type() == CV_16UC1, "8-bit raw disparity should be saved as 16-bit PNG data");
    require(output.size() == disparity.size(), "8-bit raw disparity size changed");
    require(output.at<unsigned short>(0, 0) == 255, "8-bit raw disparity value changed");
    require(output.at<unsigned short>(0, 1) == 64, "8-bit raw disparity value changed");
}

void test_preserves_16bit_raw_disparity_png_data()
{
    cv::Mat disparity(2, 3, CV_16UC1, cv::Scalar(1024));
    disparity.at<unsigned short>(0, 0) = 65520;

    cv::Mat output = stereo_examples::prepare_raw_disparity_for_png(disparity);

    require(output.type() == CV_16UC1, "16-bit raw disparity should stay 16-bit PNG data");
    require(output.size() == disparity.size(), "16-bit raw disparity size changed");
    require(output.at<unsigned short>(0, 0) == 65520, "16-bit raw disparity value changed");
    require(output.at<unsigned short>(0, 1) == 1024, "16-bit raw disparity value changed");
}

} // namespace

int main()
{
    try
    {
        test_converts_8bit_disparity_to_color();
        test_converts_16bit_subpixel_disparity_to_color();
        test_converts_8bit_raw_disparity_to_16bit_png_data();
        test_preserves_16bit_raw_disparity_png_data();
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_stereo_batch_output failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
