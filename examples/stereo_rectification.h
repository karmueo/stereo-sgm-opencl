#ifndef STEREO_RECTIFICATION_H
#define STEREO_RECTIFICATION_H

#include <string>

#include <opencv2/core/core.hpp>

namespace stereo_examples
{

struct StereoTransform
{
    cv::Mat rotation;
    cv::Mat translation;
};

StereoTransform make_opencv_stereo_transform_from_kalibr(
    const cv::Mat& rotation_cam1_from_cam0,
    const cv::Mat& translation_cam1_from_cam0);

class StereoRectifier
{
public:
    explicit StereoRectifier(const std::string& calibration_path);

    cv::Size image_size() const;
    void rectify(const cv::Mat& left,
        const cv::Mat& right,
        cv::Mat& rectified_left,
        cv::Mat& rectified_right) const;

private:
    cv::Size m_image_size;
    cv::Mat m_left_map_x;
    cv::Mat m_left_map_y;
    cv::Mat m_right_map_x;
    cv::Mat m_right_map_y;
};

} // namespace stereo_examples

#endif
