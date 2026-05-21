#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>

#include "stereo_rectification.h"

namespace
{

std::string write_test_calibration()
{
    const std::string path = "/tmp/libsgm_ocl_test_camchain.yaml";
    std::ofstream out(path.c_str());
    out << "cam0:\n"
        << "  cam_overlaps: [1]\n"
        << "  camera_model: pinhole\n"
        << "  distortion_coeffs: [0.01, -0.02, 0.001, -0.001]\n"
        << "  distortion_model: radtan\n"
        << "  intrinsics: [100.0, 101.0, 32.0, 24.0]\n"
        << "  resolution: [64, 48]\n"
        << "  rostopic: /cam0/image_raw\n"
        << "cam1:\n"
        << "  T_cn_cnm1:\n"
        << "  - [1.0, 0.0, 0.0, -0.1]\n"
        << "  - [0.0, 1.0, 0.0, 0.0]\n"
        << "  - [0.0, 0.0, 1.0, 0.0]\n"
        << "  - [0.0, 0.0, 0.0, 1.0]\n"
        << "  cam_overlaps: [0]\n"
        << "  camera_model: pinhole\n"
        << "  distortion_coeffs: [0.011, -0.021, 0.0011, -0.0011]\n"
        << "  distortion_model: radtan\n"
        << "  intrinsics: [102.0, 103.0, 33.0, 25.0]\n"
        << "  resolution: [64, 48]\n"
        << "  rostopic: /cam1/image_raw\n";
    return path;
}

void require(bool value, const std::string& message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

void test_loads_calibration()
{
    const stereo_examples::StereoRectifier rectifier(write_test_calibration());
    require(rectifier.image_size() == cv::Size(64, 48), "unexpected calibration resolution");
}

void test_rejects_wrong_input_size()
{
    const stereo_examples::StereoRectifier rectifier(write_test_calibration());
    cv::Mat left(40, 64, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::Mat right(40, 64, CV_8UC3, cv::Scalar(4, 5, 6));
    cv::Mat rectified_left;
    cv::Mat rectified_right;

    bool threw = false;
    try
    {
        rectifier.rectify(left, right, rectified_left, rectified_right);
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    require(threw, "wrong input size should throw");
}

void test_rectifies_matching_input_size()
{
    const stereo_examples::StereoRectifier rectifier(write_test_calibration());
    cv::Mat left(48, 64, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat right(48, 64, CV_8UC3, cv::Scalar(40, 50, 60));
    cv::Mat rectified_left;
    cv::Mat rectified_right;

    rectifier.rectify(left, right, rectified_left, rectified_right);
    require(rectified_left.size() == cv::Size(64, 48), "left rectified size mismatch");
    require(rectified_right.size() == cv::Size(64, 48), "right rectified size mismatch");
    require(rectified_left.type() == left.type(), "left rectified type mismatch");
    require(rectified_right.type() == right.type(), "right rectified type mismatch");
}

} // namespace

int main()
{
    try
    {
        test_loads_calibration();
        test_rejects_wrong_input_size();
        test_rectifies_matching_input_size();
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_stereo_rectification failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
