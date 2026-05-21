#include "stereo_rectification.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace
{

struct CameraCalibration
{
    std::string camera_model;
    std::string distortion_model;
    cv::Mat camera_matrix;
    cv::Mat distortion;
    cv::Size resolution;
};

struct StereoCalibration
{
    CameraCalibration cam0;
    CameraCalibration cam1;
    cv::Mat rotation;
    cv::Mat translation;
};

std::string trim(const std::string& value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();
    if (first >= last)
    {
        return std::string();
    }
    return std::string(first, last);
}

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string parse_scalar_after_colon(const std::string& line)
{
    const std::string::size_type pos = line.find(':');
    if (pos == std::string::npos)
    {
        throw std::runtime_error("missing ':' in calibration line: " + line);
    }
    return trim(line.substr(pos + 1));
}

std::vector<double> parse_number_list(const std::string& line)
{
    const std::string::size_type begin = line.find('[');
    const std::string::size_type end = line.find(']', begin == std::string::npos ? 0 : begin);
    if (begin == std::string::npos || end == std::string::npos || end <= begin)
    {
        throw std::runtime_error("missing numeric list in calibration line: " + line);
    }

    std::string values = line.substr(begin + 1, end - begin - 1);
    std::replace(values.begin(), values.end(), ',', ' ');

    std::istringstream stream(values);
    std::vector<double> result;
    double value = 0.0;
    while (stream >> value)
    {
        result.push_back(value);
    }
    if (result.empty())
    {
        throw std::runtime_error("empty numeric list in calibration line: " + line);
    }
    return result;
}

cv::Mat make_camera_matrix(const std::vector<double>& intrinsics)
{
    if (intrinsics.size() != 4)
    {
        throw std::runtime_error("intrinsics must contain [fx, fy, cx, cy]");
    }
    return (cv::Mat_<double>(3, 3) << intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);
}

cv::Mat make_distortion(const std::vector<double>& distortion)
{
    if (distortion.size() != 4)
    {
        throw std::runtime_error("distortion_coeffs must contain [k1, k2, p1, p2]");
    }
    return (cv::Mat_<double>(1, 4) << distortion[0], distortion[1], distortion[2], distortion[3]);
}

cv::Size make_resolution(const std::vector<double>& resolution)
{
    if (resolution.size() != 2)
    {
        throw std::runtime_error("resolution must contain [width, height]");
    }
    const int width = static_cast<int>(resolution[0]);
    const int height = static_cast<int>(resolution[1]);
    if (width <= 0 || height <= 0)
    {
        throw std::runtime_error("resolution must be positive");
    }
    return cv::Size(width, height);
}

void validate_camera(const CameraCalibration& camera, const std::string& name)
{
    if (camera.camera_model != "pinhole")
    {
        throw std::runtime_error(name + " camera_model must be pinhole");
    }
    if (camera.distortion_model != "radtan")
    {
        throw std::runtime_error(name + " distortion_model must be radtan");
    }
    if (camera.camera_matrix.empty())
    {
        throw std::runtime_error(name + " intrinsics are missing");
    }
    if (camera.distortion.empty())
    {
        throw std::runtime_error(name + " distortion_coeffs are missing");
    }
    if (camera.resolution.empty())
    {
        throw std::runtime_error(name + " resolution is missing");
    }
}

StereoCalibration load_kalibr_calibration(const std::string& path)
{
    std::ifstream input(path.c_str());
    if (!input)
    {
        throw std::runtime_error("failed to open calibration file: " + path);
    }

    StereoCalibration calibration;
    std::vector<std::vector<double>> transform_rows;
    CameraCalibration* current_camera = nullptr;
    bool reading_transform = false;

    std::string raw_line;
    while (std::getline(input, raw_line))
    {
        const std::string line = trim(raw_line);
        if (line.empty() || starts_with(line, "#"))
        {
            continue;
        }

        if (line == "cam0:")
        {
            current_camera = &calibration.cam0;
            reading_transform = false;
            continue;
        }
        if (line == "cam1:")
        {
            current_camera = &calibration.cam1;
            reading_transform = false;
            continue;
        }
        if (current_camera == nullptr)
        {
            continue;
        }

        if (current_camera == &calibration.cam1 && line == "T_cn_cnm1:")
        {
            reading_transform = true;
            transform_rows.clear();
            continue;
        }
        if (reading_transform && starts_with(line, "-"))
        {
            transform_rows.push_back(parse_number_list(line));
            if (transform_rows.size() == 4)
            {
                reading_transform = false;
            }
            continue;
        }
        if (starts_with(line, "camera_model:"))
        {
            current_camera->camera_model = parse_scalar_after_colon(line);
        }
        else if (starts_with(line, "distortion_model:"))
        {
            current_camera->distortion_model = parse_scalar_after_colon(line);
        }
        else if (starts_with(line, "intrinsics:"))
        {
            current_camera->camera_matrix = make_camera_matrix(parse_number_list(line));
        }
        else if (starts_with(line, "distortion_coeffs:"))
        {
            current_camera->distortion = make_distortion(parse_number_list(line));
        }
        else if (starts_with(line, "resolution:"))
        {
            current_camera->resolution = make_resolution(parse_number_list(line));
        }
    }

    validate_camera(calibration.cam0, "cam0");
    validate_camera(calibration.cam1, "cam1");
    if (calibration.cam0.resolution != calibration.cam1.resolution)
    {
        throw std::runtime_error("cam0 and cam1 resolutions must match");
    }
    if (transform_rows.size() != 4)
    {
        throw std::runtime_error("cam1.T_cn_cnm1 must contain 4 rows");
    }
    for (const auto& row : transform_rows)
    {
        if (row.size() != 4)
        {
            throw std::runtime_error("each cam1.T_cn_cnm1 row must contain 4 values");
        }
    }

    calibration.rotation = (cv::Mat_<double>(3, 3) << transform_rows[0][0], transform_rows[0][1], transform_rows[0][2],
        transform_rows[1][0], transform_rows[1][1], transform_rows[1][2],
        transform_rows[2][0], transform_rows[2][1], transform_rows[2][2]);
    calibration.translation = (cv::Mat_<double>(3, 1) << transform_rows[0][3], transform_rows[1][3], transform_rows[2][3]);
    return calibration;
}

void require_matching_size(const cv::Mat& image, const cv::Size& expected, const std::string& name)
{
    if (image.empty())
    {
        throw std::runtime_error(name + " image is empty");
    }
    if (image.size() != expected)
    {
        throw std::runtime_error(name + " image size " + std::to_string(image.cols) + "x" + std::to_string(image.rows)
            + " does not match calibration resolution " + std::to_string(expected.width) + "x" + std::to_string(expected.height));
    }
}

} // namespace

namespace stereo_examples
{

StereoTransform make_opencv_stereo_transform_from_kalibr(
    const cv::Mat& rotation_cam1_from_cam0,
    const cv::Mat& translation_cam1_from_cam0)
{
    if (rotation_cam1_from_cam0.size() != cv::Size(3, 3)
        || translation_cam1_from_cam0.rows != 3
        || translation_cam1_from_cam0.cols != 1)
    {
        throw std::runtime_error("Kalibr stereo transform must contain a 3x3 rotation and 3x1 translation");
    }

    cv::Mat rotation;
    cv::Mat translation;
    rotation_cam1_from_cam0.convertTo(rotation, CV_64F);
    translation_cam1_from_cam0.convertTo(translation, CV_64F);

    StereoTransform result;
    result.rotation = rotation.t();
    result.translation = -result.rotation * translation;
    return result;
}

StereoRectifier::StereoRectifier(const std::string& calibration_path)
{
    const StereoCalibration calibration = load_kalibr_calibration(calibration_path);
    m_image_size = calibration.cam0.resolution;
    const StereoTransform stereo_transform = make_opencv_stereo_transform_from_kalibr(
        calibration.rotation,
        calibration.translation);

    cv::Mat left_rotation;
    cv::Mat right_rotation;
    cv::Mat left_projection;
    cv::Mat right_projection;
    cv::Mat disparity_to_depth;

    cv::stereoRectify(calibration.cam0.camera_matrix,
        calibration.cam0.distortion,
        calibration.cam1.camera_matrix,
        calibration.cam1.distortion,
        m_image_size,
        stereo_transform.rotation,
        stereo_transform.translation,
        left_rotation,
        right_rotation,
        left_projection,
        right_projection,
        disparity_to_depth,
        cv::CALIB_ZERO_DISPARITY,
        1.0,
        m_image_size);

    cv::initUndistortRectifyMap(calibration.cam0.camera_matrix,
        calibration.cam0.distortion,
        left_rotation,
        left_projection,
        m_image_size,
        CV_32FC1,
        m_left_map_x,
        m_left_map_y);
    cv::initUndistortRectifyMap(calibration.cam1.camera_matrix,
        calibration.cam1.distortion,
        right_rotation,
        right_projection,
        m_image_size,
        CV_32FC1,
        m_right_map_x,
        m_right_map_y);
}

cv::Size StereoRectifier::image_size() const
{
    return m_image_size;
}

void StereoRectifier::rectify(const cv::Mat& left,
    const cv::Mat& right,
    cv::Mat& rectified_left,
    cv::Mat& rectified_right) const
{
    require_matching_size(left, m_image_size, "left");
    require_matching_size(right, m_image_size, "right");

    cv::remap(left, rectified_left, m_left_map_x, m_left_map_y, cv::INTER_LINEAR);
    cv::remap(right, rectified_right, m_right_map_x, m_right_map_y, cv::INTER_LINEAR);
}

} // namespace stereo_examples
