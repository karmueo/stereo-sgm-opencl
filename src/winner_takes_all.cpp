#include "winner_takes_all.hpp"
#include <regex>
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ocl_sgm);

//for debugging
#include <opencv2/opencv.hpp>

namespace sgm
{
namespace cl
{

template<size_t MAX_DISPARITY>
inline WinnerTakesAll<MAX_DISPARITY>::WinnerTakesAll(cl_context ctx, cl_device_id device)
    : m_cl_context(ctx)
    , m_cl_device_id(device)
    , m_tuning(ocl_runtime_config::query_tuning(device))
{

}

template<size_t MAX_DISPARITY>
WinnerTakesAll<MAX_DISPARITY>::~WinnerTakesAll()
{
    if (m_kernel)
    {
        clReleaseKernel(m_kernel);
        m_kernel = nullptr;
    }
    if (m_fast_left_kernel)
    {
        clReleaseKernel(m_fast_left_kernel);
        m_fast_left_kernel = nullptr;
    }
    if (m_fast_right_kernel)
    {
        clReleaseKernel(m_fast_right_kernel);
        m_fast_right_kernel = nullptr;
    }
}

template<size_t MAX_DISPARITY>
void WinnerTakesAll<MAX_DISPARITY>::enqueue(const DeviceBuffer<uint8_t>& src,
    int width,
    int height,
    int pitch,
    float uniqueness,
    bool subpixel,
    PathType path_type, 
    cl_command_queue stream)
{
    if (m_left_buffer.size() != static_cast<size_t>(pitch * height)) 
    {
        m_left_buffer.allocate(pitch * height);
    }
    if (m_right_buffer.size() != static_cast<size_t>(pitch * height))
    {
        m_right_buffer.allocate(pitch * height);
    }

    enqueue(m_left_buffer,
        m_right_buffer,
        src,
        width,
        height,
        pitch,
        uniqueness,
        subpixel,
        path_type,
        stream);
}

template<size_t MAX_DISPARITY>
void WinnerTakesAll<MAX_DISPARITY>::enqueue(DeviceBuffer<uint16_t>& left,
    DeviceBuffer<uint16_t>& right,
    const DeviceBuffer<uint8_t>& src,
    int width,
    int height,
    int pitch,
    float uniqueness,
    bool subpixel,
    PathType path_type,
    cl_command_queue stream)
{
    if (ocl_runtime_config::use_mali_fast_path(
            m_tuning,
            static_cast<int>(MAX_DISPARITY),
            path_type,
            subpixel))
    {
        enqueueFast(
            left,
            right,
            src,
            width,
            height,
            pitch,
            uniqueness,
            stream);
        return;
    }

    if (m_kernel == nullptr)
    {
        std::string kernel_template_types;

        //resource reading
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
        auto kernel_utility = fs.open("src/ocl/utility.cl");
        auto kernel_winner_takes_all = fs.open("src/ocl/winner_takes_all.cl");
        auto kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
            + std::string(kernel_utility.begin(), kernel_utility.end())
            + std::string(kernel_winner_takes_all.begin(), kernel_winner_takes_all.end());
        //Vertical path aggregation templates
        std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
        int NUM_PATHS = path_type == PathType::SCAN_4PATH ? 4 : 8;
        std::string kernel_NUM_PATHS = "#define NUM_PATHS " + std::to_string(NUM_PATHS) + "\n";
        std::string kernel_COMPUTE_SUBPIXEL = "#define COMPUTE_SUBPIXEL " + std::to_string(subpixel ? 1 : 0) + "\n";
        const unsigned int block_size = WARP_SIZE * m_tuning.wta_warps_per_block;
        std::string kernel_WARPS_PER_BLOCK = "#define WARPS_PER_BLOCK " + std::to_string(m_tuning.wta_warps_per_block) + "\n";
        std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(block_size) + "\n";
        std::string kernel_SUBPIXEL_SHIFT = "#define SUBPIXEL_SHIFT " + std::to_string(SubpixelShift()) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
        kernel_src = std::regex_replace(kernel_src, std::regex("@NUM_PATHS@"), kernel_NUM_PATHS);
        kernel_src = std::regex_replace(kernel_src, std::regex("@COMPUTE_SUBPIXEL@"), kernel_COMPUTE_SUBPIXEL);
        kernel_src = std::regex_replace(kernel_src, std::regex("@WARPS_PER_BLOCK@"), kernel_WARPS_PER_BLOCK);
        kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SUBPIXEL_SHIFT@"), kernel_SUBPIXEL_SHIFT);

        m_program.init(m_cl_context, m_cl_device_id,kernel_src);
        //DEBUG
        //std::cout << kernel_src << std::endl;

        m_kernel = m_program.getKernel("winner_takes_all_kernel");
    }

    const unsigned int warps_per_block = m_tuning.wta_warps_per_block;
    const unsigned int block_size = WARP_SIZE * warps_per_block;
    //setup kernels
    size_t global_size[1] = {
        (height + warps_per_block - 1) / warps_per_block * block_size
    };
    size_t local_size[1] = { block_size };


    cl_int err = clSetKernelArg(m_kernel,
        0,
        sizeof(cl_mem),
        &left.data());
    err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &right.data());
    err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &src.data());
    err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
    err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
    err = clSetKernelArg(m_kernel, 5, sizeof(pitch), &pitch);
    err = clSetKernelArg(m_kernel, 6, sizeof(uniqueness), &uniqueness);

    
    cl_event event = nullptr;
    const auto profile_start = global_ocl_profiler().kernel_start();
    err = clEnqueueNDRangeKernel(stream,
        m_kernel,
        1,
        nullptr,
        global_size,
        local_size,
        0, nullptr,
        global_ocl_profiler().event_profiling_enabled() ? &event : nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing winner_takes_all kernel");
    global_ocl_profiler().complete_kernel("winner_takes_all", stream, event, profile_start);


    //clFinish(stream);
    //cv::Mat debug(height, width, CV_16UC1);
    //clEnqueueReadBuffer(stream, left.data(), true, 0, width * height * 2, debug.data, 0, nullptr, nullptr);
    //cv::imwrite("winn_takes_all.tiff", debug);
    //cv::imshow("winner takes all debug", debug * 2048);
    //cv::waitKey(0);
}

template<size_t MAX_DISPARITY>
void WinnerTakesAll<MAX_DISPARITY>::initFast(PathType path_type)
{
    if (m_fast_left_kernel == nullptr || m_fast_right_kernel == nullptr)
    {
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
        auto kernel_winner_takes_all = fs.open("src/ocl/winner_takes_all_mali_128.cl");
        auto kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
            + std::string(kernel_winner_takes_all.begin(), kernel_winner_takes_all.end());

        std::string kernel_max_disparity = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "u\n";
        int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;
        std::string kernel_num_paths = "#define NUM_PATHS " + std::to_string(num_paths) + "u\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparity);
        kernel_src = std::regex_replace(kernel_src, std::regex("@NUM_PATHS@"), kernel_num_paths);

        m_fast_program.init(m_cl_context, m_cl_device_id, kernel_src);
        m_fast_left_kernel = m_fast_program.getKernel("winner_takes_all_left_mali_128_kernel");
        m_fast_right_kernel = m_fast_program.getKernel("winner_takes_all_right_mali_128_kernel");
    }
}

template<size_t MAX_DISPARITY>
void WinnerTakesAll<MAX_DISPARITY>::enqueueFast(DeviceBuffer<uint16_t>& left,
    DeviceBuffer<uint16_t>& right,
    const DeviceBuffer<uint8_t>& src,
    int width,
    int height,
    int pitch,
    float uniqueness,
    cl_command_queue stream)
{
    initFast(PathType::SCAN_4PATH);

    size_t global_size[2] = {
        static_cast<size_t>(width) * MAX_DISPARITY,
        static_cast<size_t>(height)
    };
    size_t local_size[2] = { MAX_DISPARITY, 1 };

    cl_int err = clSetKernelArg(m_fast_left_kernel, 0, sizeof(cl_mem), &left.data());
    err = clSetKernelArg(m_fast_left_kernel, 1, sizeof(cl_mem), &src.data());
    err = clSetKernelArg(m_fast_left_kernel, 2, sizeof(width), &width);
    err = clSetKernelArg(m_fast_left_kernel, 3, sizeof(height), &height);
    err = clSetKernelArg(m_fast_left_kernel, 4, sizeof(pitch), &pitch);
    err = clSetKernelArg(m_fast_left_kernel, 5, sizeof(uniqueness), &uniqueness);

    cl_event left_event = nullptr;
    const auto left_profile_start = global_ocl_profiler().kernel_start();
    err = clEnqueueNDRangeKernel(stream,
        m_fast_left_kernel,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr,
        global_ocl_profiler().event_profiling_enabled() ? &left_event : nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing mali winner_takes_all left kernel");
    global_ocl_profiler().complete_kernel(
        "winner_takes_all_left_mali_128",
        stream,
        left_event,
        left_profile_start);

    err = clSetKernelArg(m_fast_right_kernel, 0, sizeof(cl_mem), &right.data());
    err = clSetKernelArg(m_fast_right_kernel, 1, sizeof(cl_mem), &src.data());
    err = clSetKernelArg(m_fast_right_kernel, 2, sizeof(width), &width);
    err = clSetKernelArg(m_fast_right_kernel, 3, sizeof(height), &height);
    err = clSetKernelArg(m_fast_right_kernel, 4, sizeof(pitch), &pitch);

    cl_event right_event = nullptr;
    const auto right_profile_start = global_ocl_profiler().kernel_start();
    err = clEnqueueNDRangeKernel(stream,
        m_fast_right_kernel,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr,
        global_ocl_profiler().event_profiling_enabled() ? &right_event : nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing mali winner_takes_all right kernel");
    global_ocl_profiler().complete_kernel(
        "winner_takes_all_right_mali_128",
        stream,
        right_event,
        right_profile_start);
}

template class WinnerTakesAll<64>;
template class WinnerTakesAll<128>;
template class WinnerTakesAll<256>;

}

}
