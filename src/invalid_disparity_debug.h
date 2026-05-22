#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "libsgm_ocl/libsgm_ocl.h"

namespace sgm
{
namespace cl
{
namespace debug
{

struct InvalidDisparityDebugStats
{
    std::size_t total = 0;
    std::size_t wta_invalid = 0;
    std::size_t median_invalid = 0;
    std::size_t median_added_invalid = 0;
    std::size_t median_removed_invalid = 0;
    std::size_t final_invalid = 0;
    std::size_t mask_zero_new_invalid = 0;
    std::size_t lr_right_invalid = 0;
    std::size_t lr_diff_invalid = 0;
    std::size_t unclassified_new_invalid = 0;
};

inline uint16_t internal_invalid_disparity()
{
    return static_cast<uint16_t>(-1);
}

inline bool invalid_debug_enabled()
{
    const char* value = std::getenv("LIBSGM_OCL_DEBUG_INVALID");
    return value != nullptr && std::string(value) != "0";
}

inline std::string invalid_debug_label()
{
    const char* value = std::getenv("LIBSGM_OCL_DEBUG_LABEL");
    return value == nullptr ? std::string() : std::string(value);
}

template <typename input_type>
InvalidDisparityDebugStats compute_invalid_disparity_debug_stats(
    const uint16_t* wta_left,
    const uint16_t* median_left,
    const uint16_t* median_right,
    const uint16_t* checked_left,
    const input_type* left_pixels,
    int width,
    int height,
    int dst_pitch,
    int src_pitch,
    bool subpixel,
    int lr_max_diff)
{
    const uint16_t invalid = internal_invalid_disparity();
    InvalidDisparityDebugStats stats;
    stats.total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int disp_index = y * dst_pitch + x;
            const int src_index = y * src_pitch + x;
            const uint16_t wta = wta_left[disp_index];
            const uint16_t before = median_left[disp_index];
            const uint16_t after = checked_left[disp_index];

            const bool wta_is_invalid = wta == invalid;
            const bool before_is_invalid = before == invalid;
            const bool after_is_invalid = after == invalid;

            if (wta_is_invalid)
            {
                ++stats.wta_invalid;
            }
            if (before_is_invalid)
            {
                ++stats.median_invalid;
            }
            if (!wta_is_invalid && before_is_invalid)
            {
                ++stats.median_added_invalid;
            }
            if (wta_is_invalid && !before_is_invalid)
            {
                ++stats.median_removed_invalid;
            }
            if (after_is_invalid)
            {
                ++stats.final_invalid;
            }

            if (before_is_invalid || !after_is_invalid)
            {
                continue;
            }

            if (left_pixels[src_index] == 0)
            {
                ++stats.mask_zero_new_invalid;
                continue;
            }

            int disparity = static_cast<int>(before);
            if (subpixel)
            {
                disparity >>= SubpixelShift();
            }
            const int right_x = x - disparity;
            if (0 <= right_x && right_x < width && lr_max_diff >= 0)
            {
                const uint16_t right = median_right[y * dst_pitch + right_x];
                if (right == invalid)
                {
                    ++stats.lr_right_invalid;
                    continue;
                }
                if (std::abs(static_cast<int>(right) - disparity) > lr_max_diff)
                {
                    ++stats.lr_diff_invalid;
                    continue;
                }
            }

            ++stats.unclassified_new_invalid;
        }
    }

    return stats;
}

inline void print_invalid_disparity_debug_stats(
    const InvalidDisparityDebugStats& stats,
    const std::string& label,
    int width,
    int height)
{
    const double total = stats.total == 0 ? 1.0 : static_cast<double>(stats.total);
    const auto pct = [total](std::size_t value) {
        return 100.0 * static_cast<double>(value) / total;
    };

    std::cout << "[invalid-debug] frame=" << (label.empty() ? "-" : label)
              << " size=" << width << "x" << height
              << " total=" << stats.total << std::endl;
    std::cout << std::fixed << std::setprecision(1)
              << "[invalid-debug] wta_invalid=" << stats.wta_invalid << "(" << pct(stats.wta_invalid) << "%)"
              << " median_invalid=" << stats.median_invalid << "(" << pct(stats.median_invalid) << "%)"
              << " final_invalid=" << stats.final_invalid << "(" << pct(stats.final_invalid) << "%)" << std::endl;
    std::cout << std::fixed << std::setprecision(1)
              << "[invalid-debug] median_added_invalid=" << stats.median_added_invalid << "(" << pct(stats.median_added_invalid) << "%)"
              << " median_removed_invalid=" << stats.median_removed_invalid << "(" << pct(stats.median_removed_invalid) << "%)" << std::endl;
    std::cout << std::fixed << std::setprecision(1)
              << "[invalid-debug] mask_zero_new=" << stats.mask_zero_new_invalid << "(" << pct(stats.mask_zero_new_invalid) << "%)"
              << " lr_right_invalid=" << stats.lr_right_invalid << "(" << pct(stats.lr_right_invalid) << "%)"
              << " lr_diff_invalid=" << stats.lr_diff_invalid << "(" << pct(stats.lr_diff_invalid) << "%)"
              << " unclassified_new=" << stats.unclassified_new_invalid << "(" << pct(stats.unclassified_new_invalid) << "%)" << std::endl;
}

} // namespace debug
} // namespace cl
} // namespace sgm
