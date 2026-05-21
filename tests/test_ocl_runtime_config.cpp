#include "ocl_runtime_config.h"

#include <CL/cl.h>
#include <iostream>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}
}

int main()
{
    using namespace sgm::cl;

    require(ocl_runtime_config::is_enabled_value("1"), "1 should be enabled");
    require(ocl_runtime_config::is_enabled_value("true"), "true should be enabled");
    require(ocl_runtime_config::is_enabled_value("ON"), "ON should be enabled");
    require(ocl_runtime_config::is_enabled_value("yes"), "yes should be enabled");
    require(!ocl_runtime_config::is_enabled_value(""), "empty should be disabled");
    require(!ocl_runtime_config::is_enabled_value("0"), "0 should be disabled");
    require(!ocl_runtime_config::is_enabled_value("false"), "false should be disabled");
    require(!ocl_runtime_config::is_enabled_value("off"), "off should be disabled");

    const OclTuning generic = ocl_runtime_config::choose_tuning(
        "Generic GPU",
        16,
        256,
        CL_LOCAL);
    require(generic.census_block_size == 128, "generic census block");
    require(generic.path_block_size == 256, "generic path block");
    require(generic.wta_warps_per_block == 8, "generic WTA warps");
    require(generic.postprocess_tile_size == 16, "generic postprocess tile");

    const OclTuning mali = ocl_runtime_config::choose_tuning(
        "Mali-G52 r1 (Panfrost)",
        8,
        256,
        CL_GLOBAL);
    require(mali.census_block_size == 64, "mali census block");
    require(mali.path_block_size == 128, "mali path block");
    require(mali.wta_warps_per_block == 4, "mali WTA warps");
    require(mali.postprocess_tile_size == 8, "mali postprocess tile");
    require(!mali.default_subpixel, "mali realtime default subpixel");
    require(!mali.use_mali_vertical_128, "mali vertical rewrite default off");
    require(!mali.use_mali_wta_128_4path, "mali WTA rewrite default off");
    require(!ocl_runtime_config::use_mali_fast_path(mali, 128, PathType::SCAN_4PATH, false), "mali rewrite default fallback");
    OclTuning enabled_mali = mali;
    ocl_runtime_config::enable_mali_rewrite_kernels(enabled_mali);
    require(enabled_mali.use_mali_vertical_128, "enabled mali vertical rewrite");
    require(enabled_mali.use_mali_wta_128_4path, "enabled mali WTA rewrite");
    require(ocl_runtime_config::use_mali_fast_path(enabled_mali, 128, PathType::SCAN_4PATH, false), "enabled mali 128/4path/non-subpixel fast path");
    require(!ocl_runtime_config::use_mali_fast_path(enabled_mali, 64, PathType::SCAN_4PATH, false), "mali 64 fallback");
    require(!ocl_runtime_config::use_mali_fast_path(enabled_mali, 256, PathType::SCAN_4PATH, false), "mali 256 fallback");
    require(!ocl_runtime_config::use_mali_fast_path(enabled_mali, 128, PathType::SCAN_8PATH, false), "mali 8path fallback");
    require(!ocl_runtime_config::use_mali_fast_path(enabled_mali, 128, PathType::SCAN_4PATH, true), "mali subpixel fallback");
    require(!ocl_runtime_config::use_mali_fast_path(generic, 128, PathType::SCAN_4PATH, false), "generic fallback");
    OclTuning disabled_mali = enabled_mali;
    ocl_runtime_config::disable_mali_fast_paths(disabled_mali);
    require(!ocl_runtime_config::use_mali_fast_path(disabled_mali, 128, PathType::SCAN_4PATH, false), "disabled mali fallback");

    return 0;
}
