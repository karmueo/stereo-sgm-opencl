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

#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <libsgm_ocl/libsgm_ocl.h>
#include <iomanip>
#include <sstream>
#include "stereo_rectification.h"

void context_error_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data);
std::tuple<cl_context, cl_device_id> initCLCTX(int platform_idx, int device_idx);


int main(int argc, char* argv[])
{
    std::string keys =
    "{ h help | | Print this message }"
    "{ @img_source_left | | Left images }"
    "{ @img_source_right |  | Right images }"
    "{ md max_disparity | 128 | Maximum disparity }"
    "{ sp subpixel | true | Compute subpixel accuracy }"
    "{ platform_idx | 0 | OpenCL plarform index }"
    "{ device_idx | 0 | OpenCL device index }"
    "{ np num_path | 4 | Num path to optimize, 4 or 8 }"
    "{ no_display | false | Disable OpenCV display windows }"
    "{ output | | Optional output disparity image path }"
    "{ calib | stereo-camchain.yaml | Stereo calibration YAML path }"
    "{ no_rectify | false | Disable stereo rectification }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }
    std::string left_filename_fmt, right_filename_fmt;
    left_filename_fmt = parser.get<std::string>(0);
    right_filename_fmt = parser.get<std::string>(1);
    int disp_size = parser.get<int>("max_disparity");
    std::string output_path = parser.get<std::string>("output");
    bool display_enabled = !parser.get<bool>("no_display");
#ifndef _WIN32
    display_enabled = display_enabled && (std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr);
#endif
    if (!display_enabled)
    {
        if (output_path.empty())
        {
            std::cout << "Display disabled; use --output to save the disparity image." << std::endl;
        }
        else
        {
            std::cout << "Display disabled; saving disparity image to: " << output_path << std::endl;
        }
    }

    cv::VideoCapture left_capture(left_filename_fmt);
    cv::VideoCapture right_capture(right_filename_fmt);
    if (!left_capture.isOpened())
    {
        std::cout << "Failed to open image stream: " << left_filename_fmt << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (!right_capture.isOpened())
    {
        std::cout << "Failed to open image stream: " << right_filename_fmt << std::endl;
        std::exit(EXIT_FAILURE);
    }
    left_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
    right_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);

    std::unique_ptr<stereo_examples::StereoRectifier> rectifier;
    try
    {
        if (!parser.get<bool>("no_rectify"))
        {
            rectifier.reset(new stereo_examples::StereoRectifier(parser.get<std::string>("calib")));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    auto convertTo4size = [](const cv::Mat& m) -> cv::Mat {
        int new_size_x = (m.cols / 4) * 4;
        int new_size_y = (m.rows / 4) * 4;
        cv::Mat ret = m(cv::Rect(0, 0, new_size_x, new_size_y)).clone();
        return ret;
    };

    auto prepare_pair = [&](const cv::Mat& raw_left, const cv::Mat& raw_right, cv::Mat& left_gray, cv::Mat& right_gray) {
        if (raw_left.empty())
        {
            throw std::runtime_error("failed to read left image stream");
        }
        if (raw_right.empty())
        {
            throw std::runtime_error("failed to read right image stream");
        }

        cv::Mat prepared_left = raw_left;
        cv::Mat prepared_right = raw_right;
        if (rectifier)
        {
            rectifier->rectify(raw_left, raw_right, prepared_left, prepared_right);
        }

        prepared_left = convertTo4size(prepared_left);
        prepared_right = convertTo4size(prepared_right);
        if (prepared_left.empty() || prepared_right.empty())
        {
            throw std::runtime_error("image is too small after crop");
        }
        if (prepared_left.size() != prepared_right.size() || prepared_left.type() != prepared_right.type())
        {
            throw std::runtime_error("mismatch input image size or type");
        }

        if (prepared_left.channels() != 1)
        {
            cv::cvtColor(prepared_left, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(prepared_right, right_gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            left_gray = prepared_left;
            right_gray = prepared_right;
        }
    };

    cv::Mat img1c, img2c;
    left_capture >> img1c;
    right_capture >> img2c;

    cv::Mat left, right;
    try
    {
        prepare_pair(img1c, img2c, left, right);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (left.size() != right.size() || left.type() != right.type())
    {
        std::cerr << "mismatch input image size" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cv::Size img_size = left.size();

    int width = left.cols;
    int height = left.rows;

    if (width * height == 0)
    {
        std::cout << "Wrong input size: " << width << ", " << height << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cl_context cl_ctx;
    cl_device_id cl_device;
    int platform_idx = parser.get<int>("platform_idx");
    int device_idx = parser.get<int>("device_idx");
    try
    {
        std::tie(cl_ctx, cl_device) = initCLCTX(platform_idx, device_idx);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    cl_int queue_err = CL_SUCCESS;
    cl_command_queue cl_queue = clCreateCommandQueue(cl_ctx, cl_device, 0, &queue_err);
    if (queue_err != CL_SUCCESS || cl_queue == nullptr)
    {
        std::cerr << "Error creating command queue: " << queue_err << std::endl;
        clReleaseContext(cl_ctx);
        return EXIT_FAILURE;
    }

    sgm::cl::Parameters params;
    int input_depth = 8;
    params.subpixel = parser.get<bool>("subpixel");
    const int output_depth = (disp_size == 256 || params.subpixel) ? 16 : 8;
    params.path_type = parser.get<int>("num_path") == 8 ? sgm::cl::PathType::SCAN_8PATH : sgm::cl::PathType::SCAN_4PATH;
    params.uniqueness = 0.95f;

    {
        sgm::cl::StereoSGM<uint8_t> ssgm(width,
            height,
            disp_size,
            output_depth,
            cl_ctx,
            cl_device,
            params);

        bool should_close = false;
        int disp_type = output_depth == 8 ? CV_8UC1 : CV_16UC1;

        cv::Mat disp(img_size, disp_type), disp_color, disp_8u;
        cl_mem d_left, d_right, d_disp;
        d_left = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height, nullptr, nullptr);
        d_right = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height, nullptr, nullptr);
        d_disp = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height * output_depth / 8, nullptr, nullptr);

        while ((!should_close))
        {
            if (img1c.empty())
            {
                std::cout << "Failed to read left image stream!" << std::endl;
                break;
            }
            if (img2c.empty())
            {
                std::cout << "Failed to read right image stream!" << std::endl;
                break;
            }

            try
            {
                prepare_pair(img1c, img2c, left, right);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                break;
            }
            if (left.size() != img_size || right.size() != img_size || left.type() != right.type())
            {
                std::cerr << "mismatch input image size or type" << std::endl;
                break;
            }

            clEnqueueWriteBuffer(cl_queue, d_left, true, 0, width * height, left.data, 0, nullptr, nullptr);
            clEnqueueWriteBuffer(cl_queue, d_right, true, 0, width * height, right.data, 0, nullptr, nullptr);

            auto t = std::chrono::steady_clock::now();
            //ssgm.execute(left.data, right.data, reinterpret_cast<uint16_t*>(disp.data));
            ssgm.execute(d_left, d_right, d_disp);
            const double dur_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
            clEnqueueReadBuffer(cl_queue, d_disp, true, 0, width * height * output_depth / 8, disp.data, 0, nullptr, nullptr);


            cv::Mat disparity_8u, disparity_color;
            disp.convertTo(disparity_8u, CV_8U, 255. / (disp_size * (params.subpixel ? 16 : 1)));
            cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
            const int invalid_disp = output_depth == 8
                ? static_cast<uint8_t>(ssgm.get_invalid_disparity())
                : static_cast<uint16_t>(ssgm.get_invalid_disparity());
            disparity_color.setTo(cv::Scalar(0, 0, 0), disp == invalid_disp);
            const double fps = dur_ms > 0.0 ? 1000.0 / dur_ms : 0.0;
            std::cout << "SGM execution time: " << dur_ms << " ms, " << fps << " FPS" << std::endl;
            cv::putText(disparity_color, "sgm execution time: " + std::to_string(dur_ms) + "[msec] " + std::to_string(fps) + "[FPS]",
                cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));


            if (!output_path.empty())
            {
                if (cv::imwrite(output_path, disparity_color))
                {
                    std::cout << "Saved disparity image: " << output_path << std::endl;
                }
                else
                {
                    std::cerr << "Failed to save disparity image: " << output_path << std::endl;
                }
            }

            if (display_enabled)
            {
                cv::imshow("left imagep", left);
                cv::imshow("disp", disparity_color);

                int key = cv::waitKey(1);
                if (key == 27)
                {
                    should_close = true;
                }
            }

            if (!left_capture.read(img1c) || !right_capture.read(img2c))
            {
                break;
            }
        }
        clReleaseMemObject(d_left);
        clReleaseMemObject(d_right);
        clReleaseMemObject(d_disp);
    }
    clReleaseCommandQueue(cl_queue);
    clReleaseDevice(cl_device);
    clReleaseContext(cl_ctx);
    return 0;
}


std::tuple<cl_context, cl_device_id> initCLCTX(int platform_idx, int device_idx)
{
    cl_uint num_platform = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platform);
    if (err != CL_SUCCESS || num_platform == 0)
    {
        throw std::runtime_error("no OpenCL platforms found");
    }
    if (platform_idx < 0 || static_cast<cl_uint>(platform_idx) >= num_platform)
    {
        throw std::runtime_error("platform_idx is out of range");
    }

    std::vector<cl_platform_id> platform_ids(num_platform);
    err = clGetPlatformIDs(num_platform, platform_ids.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("failed to enumerate OpenCL platforms, error " + std::to_string(err));
    }

    cl_platform_id platform = platform_ids[platform_idx];
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

    std::vector<cl_device_id> cl_devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, cl_devices.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("failed to enumerate OpenCL GPU devices, error " + std::to_string(err));
    }

    cl_device_id cl_device = cl_devices[device_idx];
    cl_context cl_ctx = clCreateContext(nullptr, 1, &cl_devices[device_idx], context_error_callback, NULL, &err);

    if (err != CL_SUCCESS || cl_ctx == nullptr)
    {
        throw std::runtime_error("failed to create OpenCL context, error " + std::to_string(err));
    }
    {
        size_t name_size_in_bytes;
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &name_size_in_bytes);
        std::string platform_name;
        platform_name.resize(name_size_in_bytes);
        clGetPlatformInfo(platform, CL_PLATFORM_NAME,
            name_size_in_bytes,
            (void*)platform_name.data(), nullptr);
        if (!platform_name.empty() && platform_name.back() == '\0')
        {
            platform_name.pop_back();
        }
        std::cout << "Platform name: " << platform_name << std::endl;
    }
    {
        size_t name_size_in_bytes;
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME, 0, nullptr, &name_size_in_bytes);
        std::string dev_name;
        dev_name.resize(name_size_in_bytes);
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME,
            name_size_in_bytes,
            (void*)dev_name.data(), nullptr);
        if (!dev_name.empty() && dev_name.back() == '\0')
        {
            dev_name.pop_back();
        }
        std::cout << "Device name: " << dev_name << std::endl;
    }
    return std::make_tuple(cl_ctx, cl_device);
}

void context_error_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data)
{
    std::cout << "opencl error : " << errinfo << std::endl;
}
