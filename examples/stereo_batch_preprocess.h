#ifndef STEREO_BATCH_PREPROCESS_H
#define STEREO_BATCH_PREPROCESS_H

#include <cmath>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace stereo_examples
{

inline void validate_sgm_scale(double scale)
{
    if (!std::isfinite(scale) || scale <= 0.0 || scale > 1.0)
    {
        throw std::runtime_error("scale must be greater than 0.0 and less than or equal to 1.0");
    }
}

inline cv::Mat scale_image_for_sgm(const cv::Mat& image, double scale)
{
    validate_sgm_scale(scale);
    if (scale == 1.0)
    {
        return image;
    }

    const int scaled_width = static_cast<int>(std::round(image.cols * scale));
    const int scaled_height = static_cast<int>(std::round(image.rows * scale));
    if (scaled_width <= 0 || scaled_height <= 0)
    {
        throw std::runtime_error("scaled image size must be positive");
    }

    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(scaled_width, scaled_height), 0.0, 0.0, cv::INTER_AREA);
    return scaled;
}

} // namespace stereo_examples

#endif
