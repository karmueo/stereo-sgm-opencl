#pragma once

#include "libsgm_ocl/types.h"

#include <CL/cl.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace sgm
{
namespace cl
{

struct OclTuning
{
    unsigned int census_block_size = 128;
    unsigned int path_block_size = 256;
    unsigned int wta_warps_per_block = 8;
    unsigned int postprocess_tile_size = 16;
    bool default_subpixel = false;
    bool mali_g52_profile = false;
    bool use_mali_vertical_128 = false;
    bool use_mali_wta_128_4path = false;
};

namespace ocl_runtime_config
{

inline std::string normalized(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool is_enabled_value(const std::string& value)
{
    const std::string v = normalized(value);
    return v == "1" || v == "true" || v == "on" || v == "yes";
}

inline bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && is_enabled_value(value);
}

inline void disable_mali_fast_paths(OclTuning& tuning)
{
    tuning.use_mali_vertical_128 = false;
    tuning.use_mali_wta_128_4path = false;
}

inline void enable_mali_rewrite_kernels(OclTuning& tuning)
{
    if (tuning.mali_g52_profile)
    {
        tuning.use_mali_vertical_128 = true;
        tuning.use_mali_wta_128_4path = true;
    }
}

inline bool contains(std::string value, const std::string& needle)
{
    value = normalized(value);
    return value.find(normalized(needle)) != std::string::npos;
}

inline OclTuning choose_tuning(
    const std::string& device_name,
    cl_uint preferred_work_group_multiple,
    size_t max_work_group_size,
    cl_device_local_mem_type local_mem_type)
{
    OclTuning tuning;
    const bool is_mali_g52 = contains(device_name, "mali-g52")
        || (contains(device_name, "mali") && contains(device_name, "g52"));

    if (is_mali_g52
        && preferred_work_group_multiple <= 8
        && max_work_group_size >= 128
        && local_mem_type == CL_GLOBAL)
    {
        tuning.census_block_size = 64;
        tuning.path_block_size = 128;
        tuning.wta_warps_per_block = 4;
        tuning.postprocess_tile_size = 8;
        tuning.mali_g52_profile = true;
    }

    return tuning;
}

inline OclTuning query_tuning(cl_device_id device)
{
    if (device == nullptr)
    {
        return OclTuning();
    }

    size_t name_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &name_size);
    std::string device_name(name_size, '\0');
    if (name_size > 0)
    {
        clGetDeviceInfo(device, CL_DEVICE_NAME, name_size, &device_name[0], nullptr);
        if (!device_name.empty() && device_name.back() == '\0')
        {
            device_name.pop_back();
        }
    }

    cl_uint preferred_multiple = 8;
    size_t max_work_group_size = 0;
    cl_device_local_mem_type local_mem_type = CL_LOCAL;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, nullptr);

    OclTuning tuning = choose_tuning(device_name, preferred_multiple, max_work_group_size, local_mem_type);
    if (env_enabled("LIBSGM_OCL_ENABLE_MALI_REWRITE"))
    {
        enable_mali_rewrite_kernels(tuning);
    }
    if (env_enabled("LIBSGM_OCL_DISABLE_MALI_FAST_PATH"))
    {
        disable_mali_fast_paths(tuning);
    }
    return tuning;
}

inline bool use_mali_fast_path(
    const OclTuning& tuning,
    int max_disparity,
    PathType path_type,
    bool subpixel)
{
    return tuning.mali_g52_profile
        && tuning.use_mali_vertical_128
        && tuning.use_mali_wta_128_4path
        && max_disparity == 128
        && path_type == PathType::SCAN_4PATH
        && !subpixel;
}

} // namespace ocl_runtime_config
} // namespace cl
} // namespace sgm
