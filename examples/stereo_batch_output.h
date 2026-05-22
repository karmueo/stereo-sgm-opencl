#ifndef STEREO_BATCH_OUTPUT_H
#define STEREO_BATCH_OUTPUT_H

#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace stereo_examples
{

inline cv::Mat prepare_disparity_for_png(
    const cv::Mat& disparity,
    int max_disparity,
    bool subpixel,
    int invalid_disparity)
{
    if (disparity.type() != CV_8UC1 && disparity.type() != CV_16UC1)
    {
        throw std::runtime_error("disparity image must be CV_8UC1 or CV_16UC1");
    }
    if (max_disparity <= 0)
    {
        throw std::runtime_error("max_disparity must be positive");
    }

    cv::Mat disparity_8u;
    cv::Mat disparity_color;
    disparity.convertTo(disparity_8u, CV_8U, 255.0 / (max_disparity * (subpixel ? 16 : 1)));
    cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);

    if (disparity.type() == CV_8UC1)
    {
        disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == static_cast<unsigned char>(invalid_disparity));
    }
    else
    {
        disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == static_cast<unsigned short>(invalid_disparity));
    }
    return disparity_color;
}

inline cv::Mat prepare_raw_disparity_for_png(const cv::Mat& disparity)
{
    if (disparity.type() != CV_8UC1 && disparity.type() != CV_16UC1)
    {
        throw std::runtime_error("raw disparity image must be CV_8UC1 or CV_16UC1");
    }

    cv::Mat raw_disparity;
    if (disparity.type() == CV_8UC1)
    {
        disparity.convertTo(raw_disparity, CV_16U);
    }
    else
    {
        raw_disparity = disparity.clone();
    }
    return raw_disparity;
}

} // namespace stereo_examples

#endif
