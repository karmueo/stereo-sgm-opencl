/*
Copyright 2016 fixstars

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <CL/cl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <libsgm_ocl/libsgm_ocl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stereo_rectification.h"
#include "stereo_batch_output.h"
#include "stereo_batch_preprocess.h"
#include "ocl_profiler.h"

namespace
{

struct StereoPair
{
    std::string suffix_with_ext;
    std::string output_stem;
    std::string left_path;
    std::string right_path;
};

struct LoadedPair
{
    StereoPair pair;
    cv::Mat left;
    cv::Mat right;
};

struct ClContext
{
    cl_context context = nullptr;
    cl_device_id device = nullptr;
};

std::string join_path(const std::string& base, const std::string& name)
{
    if (base.empty() || base.back() == '/')
    {
        return base + name;
    }
    return base + "/" + name;
}

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size()
        && value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size()
        && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string remove_extension(const std::string& filename)
{
    const std::string::size_type pos = filename.find_last_of('.');
    if (pos == std::string::npos)
    {
        return filename;
    }
    return filename.substr(0, pos);
}

bool is_regular_file(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

void create_directories(const std::string& path)
{
    if (path.empty())
    {
        throw std::runtime_error("output_dir is empty");
    }

    std::string partial;
    std::size_t pos = 0;
    if (path[0] == '/')
    {
        partial = "/";
        pos = 1;
    }

    while (pos <= path.size())
    {
        const std::size_t next = path.find('/', pos);
        const std::string part = path.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        if (!part.empty())
        {
            if (!partial.empty() && partial.back() != '/')
            {
                partial += "/";
            }
            partial += part;

            struct stat st;
            if (stat(partial.c_str(), &st) == 0)
            {
                if (!S_ISDIR(st.st_mode))
                {
                    throw std::runtime_error(partial + " exists but is not a directory");
                }
            }
            else if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
            {
                throw std::runtime_error("failed to create directory " + partial + ": " + std::strerror(errno));
            }
        }

        if (next == std::string::npos)
        {
            break;
        }
        pos = next + 1;
    }
}

std::vector<StereoPair> find_stereo_pairs(
    const std::string& input_dir,
    const std::string& left_prefix,
    const std::string& right_prefix)
{
    DIR* dir = opendir(input_dir.c_str());
    if (dir == nullptr)
    {
        throw std::runtime_error("failed to open input_dir " + input_dir + ": " + std::strerror(errno));
    }

    std::map<std::string, std::string> left_by_suffix;
    std::map<std::string, std::string> right_by_suffix;

    while (dirent* ent = readdir(dir))
    {
        const std::string filename(ent->d_name);
        if (filename == "." || filename == ".." || !ends_with(filename, ".png"))
        {
            continue;
        }

        const std::string path = join_path(input_dir, filename);
        if (!is_regular_file(path))
        {
            continue;
        }

        if (starts_with(filename, left_prefix))
        {
            left_by_suffix[filename.substr(left_prefix.size())] = path;
        }
        else if (starts_with(filename, right_prefix))
        {
            right_by_suffix[filename.substr(right_prefix.size())] = path;
        }
    }
    closedir(dir);

    std::vector<StereoPair> pairs;
    for (const auto& item : left_by_suffix)
    {
        const auto right = right_by_suffix.find(item.first);
        if (right == right_by_suffix.end())
        {
            continue;
        }

        StereoPair pair;
        pair.suffix_with_ext = item.first;
        pair.output_stem = remove_extension(item.first);
        pair.left_path = item.second;
        pair.right_path = right->second;
        pairs.push_back(pair);
    }

    return pairs;
}

cv::Mat crop_to_multiple_of_4(const cv::Mat& image)
{
    const int cols = (image.cols / 4) * 4;
    const int rows = (image.rows / 4) * 4;
    if (cols <= 0 || rows <= 0)
    {
        return cv::Mat();
    }
    return image(cv::Rect(0, 0, cols, rows)).clone();
}

std::vector<LoadedPair> load_pairs(
    const std::vector<StereoPair>& pairs,
    const stereo_examples::StereoRectifier* rectifier,
    double scale)
{
    std::vector<LoadedPair> loaded;
    loaded.reserve(pairs.size());

    for (const auto& pair : pairs)
    {
        cv::Mat left = cv::imread(pair.left_path, cv::IMREAD_UNCHANGED);
        cv::Mat right = cv::imread(pair.right_path, cv::IMREAD_UNCHANGED);
        if (left.empty())
        {
            throw std::runtime_error("failed to read left image: " + pair.left_path);
        }
        if (right.empty())
        {
            throw std::runtime_error("failed to read right image: " + pair.right_path);
        }

        if (rectifier != nullptr)
        {
            cv::Mat rectified_left;
            cv::Mat rectified_right;
            rectifier->rectify(left, right, rectified_left, rectified_right);
            left = rectified_left;
            right = rectified_right;
        }

        left = stereo_examples::scale_image_for_sgm(left, scale);
        right = stereo_examples::scale_image_for_sgm(right, scale);

        left = crop_to_multiple_of_4(left);
        right = crop_to_multiple_of_4(right);
        if (left.empty() || right.empty())
        {
            throw std::runtime_error("image is too small after crop: " + pair.suffix_with_ext);
        }
        if (left.size() != right.size() || left.type() != right.type())
        {
            throw std::runtime_error("left/right image size or type mismatch: " + pair.suffix_with_ext);
        }

        LoadedPair item;
        item.pair = pair;
        item.left = left;
        item.right = right;
        loaded.push_back(item);
    }

    return loaded;
}

void context_error_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data)
{
    (void)private_info;
    (void)cb;
    (void)user_data;
    std::cerr << "OpenCL error: " << errinfo << std::endl;
}

std::string get_platform_name(cl_platform_id platform)
{
    size_t size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &size);
    std::string name(size, '\0');
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, &name[0], nullptr);
    if (!name.empty() && name.back() == '\0')
    {
        name.pop_back();
    }
    return name;
}

std::string get_device_name(cl_device_id device)
{
    size_t size = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size);
    std::string name(size, '\0');
    clGetDeviceInfo(device, CL_DEVICE_NAME, size, &name[0], nullptr);
    if (!name.empty() && name.back() == '\0')
    {
        name.pop_back();
    }
    return name;
}

ClContext init_cl_context(int platform_idx, int device_idx)
{
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0)
    {
        throw std::runtime_error("no OpenCL platforms found; install/enable the RK GPU OpenCL ICD");
    }
    if (platform_idx < 0 || static_cast<cl_uint>(platform_idx) >= num_platforms)
    {
        throw std::runtime_error("platform_idx is out of range");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[platform_idx];

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0)
    {
        throw std::runtime_error("no GPU OpenCL devices found on platform " + std::to_string(platform_idx));
    }
    if (device_idx < 0 || static_cast<cl_uint>(device_idx) >= num_devices)
    {
        throw std::runtime_error("device_idx is out of range");
    }

    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

    ClContext result;
    result.device = devices[device_idx];
    result.context = clCreateContext(nullptr, 1, &result.device, context_error_callback, nullptr, &err);
    if (err != CL_SUCCESS || result.context == nullptr)
    {
        throw std::runtime_error("failed to create OpenCL context, error " + std::to_string(err));
    }

    std::cout << "Platform name: " << get_platform_name(platform) << std::endl;
    std::cout << "Device name: " << get_device_name(result.device) << std::endl;
    return result;
}

void check_cl(cl_int err, const std::string& what)
{
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error(what + " failed, OpenCL error " + std::to_string(err));
    }
}

cv::Mat to_gray(const cv::Mat& image)
{
    if (image.channels() == 1)
    {
        return image;
    }

    cv::Mat gray;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::runtime_error("unsupported image channel count: " + std::to_string(image.channels()));
    }
    return gray;
}

} // namespace

int main(int argc, char* argv[])
{
    const std::string keys =
        "{ h help | | Print this message }"
        "{ input_dir | /home/sjgd/Public/stereo | Directory containing left/right images }"
        "{ output_dir | | Directory for output disparity PNG files }"
        "{ left_prefix | left_ | Left image filename prefix }"
        "{ right_prefix | right_ | Right image filename prefix }"
        "{ md max_disparity | 128 | Maximum disparity, one of 64, 128, 256 }"
        "{ sp subpixel | false | Compute subpixel accuracy }"
        "{ platform_idx | 0 | OpenCL platform index }"
        "{ device_idx | 0 | OpenCL GPU device index }"
        "{ np num_path | 4 | Num path to optimize, 4 or 8 }"
        "{ calib | stereo-camchain.yaml | Stereo calibration YAML path }"
        "{ no_rectify | false | Disable stereo rectification }"
        "{ scale | 1.0 | Image scale factor before SGM computation, > 0 and <= 1 }"
        "{ profile | false | Print OpenCL kernel profiling }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    try
    {
        const std::string input_dir = parser.get<std::string>("input_dir");
        const std::string output_dir = parser.get<std::string>("output_dir");
        const std::string left_prefix = parser.get<std::string>("left_prefix");
        const std::string right_prefix = parser.get<std::string>("right_prefix");
        const int disp_size = parser.get<int>("max_disparity");
        const int num_path = parser.get<int>("num_path");
        const double scale = parser.get<double>("scale");
        const bool profile_enabled = parser.get<bool>("profile");
        if (profile_enabled)
        {
#ifdef _WIN32
            _putenv_s("LIBSGM_OCL_PROFILE", "1");
#else
            setenv("LIBSGM_OCL_PROFILE", "1", 1);
#endif
        }

        if (output_dir.empty())
        {
            throw std::runtime_error("output_dir is required");
        }
        if (disp_size != 64 && disp_size != 128 && disp_size != 256)
        {
            throw std::runtime_error("max_disparity must be one of 64, 128, 256");
        }
        if (num_path != 4 && num_path != 8)
        {
            throw std::runtime_error("num_path must be 4 or 8");
        }
        stereo_examples::validate_sgm_scale(scale);

        const std::vector<StereoPair> pairs = find_stereo_pairs(input_dir, left_prefix, right_prefix);
        if (pairs.empty())
        {
            throw std::runtime_error("no matched stereo pairs found in " + input_dir);
        }

        std::cout << "Loading " << pairs.size() << " stereo pairs from: " << input_dir << std::endl;
        std::unique_ptr<stereo_examples::StereoRectifier> rectifier;
        if (!parser.get<bool>("no_rectify"))
        {
            rectifier.reset(new stereo_examples::StereoRectifier(parser.get<std::string>("calib")));
        }
        std::vector<LoadedPair> loaded = load_pairs(pairs, rectifier.get(), scale);
        create_directories(output_dir);

        const cv::Size img_size = loaded.front().left.size();
        const int input_type = loaded.front().left.type();
        for (const auto& item : loaded)
        {
            if (item.left.size() != img_size || item.right.size() != img_size)
            {
                throw std::runtime_error("all images must have the same cropped size: " + item.pair.suffix_with_ext);
            }
            if (item.left.type() != input_type || item.right.type() != input_type)
            {
                throw std::runtime_error("all images must have the same type: " + item.pair.suffix_with_ext);
            }
        }

        ClContext cl = init_cl_context(parser.get<int>("platform_idx"), parser.get<int>("device_idx"));
        cl_int err = CL_SUCCESS;
        cl_command_queue queue = sgm::cl::create_ocl_command_queue(cl.context, cl.device, &err);
        check_cl(err, "clCreateCommandQueue");

        sgm::cl::Parameters params;
        params.subpixel = parser.get<bool>("subpixel");
        params.path_type = num_path == 8 ? sgm::cl::PathType::SCAN_8PATH : sgm::cl::PathType::SCAN_4PATH;
        params.uniqueness = 0.95f;

        const int output_depth = (disp_size == 256 || params.subpixel) ? 16 : 8;
        const int disp_type = output_depth == 8 ? CV_8UC1 : CV_16UC1;
        const size_t image_bytes = static_cast<size_t>(img_size.width) * img_size.height;
        const size_t disp_bytes = image_bytes * output_depth / 8;

        sgm::cl::StereoSGM<uint8_t> ssgm(
            img_size.width,
            img_size.height,
            disp_size,
            output_depth,
            cl.context,
            cl.device,
            params);

        cl_mem d_left = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, image_bytes, nullptr, &err);
        check_cl(err, "clCreateBuffer(left)");
        cl_mem d_right = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, image_bytes, nullptr, &err);
        check_cl(err, "clCreateBuffer(right)");
        cl_mem d_disp = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, disp_bytes, nullptr, &err);
        check_cl(err, "clCreateBuffer(disparity)");

        double total_processing_ms = 0.0;
        int processed = 0;
        std::cout << "Processing size: " << img_size.width << "x" << img_size.height
                  << ", output_depth=" << output_depth
                  << ", scale=" << scale
                  << ", pairs=" << loaded.size() << std::endl;

        for (const auto& item : loaded)
        {
            cv::Mat disp(img_size, disp_type);
            cv::Mat disp_to_save;

            const auto processing_start = std::chrono::steady_clock::now();
            cv::Mat left_gray = to_gray(item.left);
            cv::Mat right_gray = to_gray(item.right);
            if (!left_gray.isContinuous())
            {
                left_gray = left_gray.clone();
            }
            if (!right_gray.isContinuous())
            {
                right_gray = right_gray.clone();
            }

            check_cl(clEnqueueWriteBuffer(queue, d_left, CL_TRUE, 0, image_bytes, left_gray.data, 0, nullptr, nullptr),
                "clEnqueueWriteBuffer(left)");
            check_cl(clEnqueueWriteBuffer(queue, d_right, CL_TRUE, 0, image_bytes, right_gray.data, 0, nullptr, nullptr),
                "clEnqueueWriteBuffer(right)");
            ssgm.execute(d_left, d_right, d_disp);
            check_cl(clEnqueueReadBuffer(queue, d_disp, CL_TRUE, 0, disp_bytes, disp.data, 0, nullptr, nullptr),
                "clEnqueueReadBuffer(disparity)");
            disp_to_save = stereo_examples::prepare_disparity_for_png(
                disp,
                disp_size,
                params.subpixel,
                ssgm.get_invalid_disparity());
            const auto processing_end = std::chrono::steady_clock::now();

            const double frame_ms = std::chrono::duration<double, std::milli>(processing_end - processing_start).count();
            total_processing_ms += frame_ms;
            ++processed;

            const std::string output_path = join_path(output_dir, "disparity_" + item.pair.output_stem + ".png");
            if (!cv::imwrite(output_path, disp_to_save))
            {
                throw std::runtime_error("failed to write disparity image: " + output_path);
            }

            const double frame_fps = frame_ms > 0.0 ? 1000.0 / frame_ms : 0.0;
            std::cout << "Processed " << item.pair.suffix_with_ext << ": "
                      << std::fixed << std::setprecision(3) << frame_ms << " ms, "
                      << frame_fps << " FPS" << std::endl;
        }

        clReleaseMemObject(d_disp);
        clReleaseMemObject(d_right);
        clReleaseMemObject(d_left);
        clReleaseCommandQueue(queue);
        clReleaseContext(cl.context);

        const double avg_ms = processed > 0 ? total_processing_ms / processed : 0.0;
        const double avg_fps = total_processing_ms > 0.0 ? processed * 1000.0 / total_processing_ms : 0.0;
        std::cout << "Processed frames: " << processed << std::endl;
        std::cout << "Total processing time (no IO): " << std::fixed << std::setprecision(3)
                  << total_processing_ms << " ms" << std::endl;
        std::cout << "Average processing time: " << avg_ms << " ms/frame" << std::endl;
        std::cout << "Average FPS (no IO): " << avg_fps << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
