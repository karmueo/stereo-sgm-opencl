#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "invalid_disparity_debug.h"

namespace
{

void require(bool value, const std::string& message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

void test_classifies_invalid_sources()
{
    const int width = 4;
    const int height = 2;
    const int pitch = 4;
    const uint16_t invalid = sgm::cl::debug::internal_invalid_disparity();

    const std::vector<uint16_t> wta_left = {
        invalid, 1, 1, 1,
        1, 1, 1, 1
    };
    const std::vector<uint16_t> median_left = {
        invalid, 1, 1, 1,
        invalid, 1, 1, 1
    };
    const std::vector<uint16_t> median_right = {
        0, invalid, 4, 1,
        0, 1, 2, 1
    };
    const std::vector<uint16_t> checked_left = {
        invalid, invalid, invalid, invalid,
        invalid, 1, 1, 1
    };
    const std::vector<uint8_t> left_pixels = {
        8, 0, 8, 8,
        8, 8, 8, 8
    };

    const sgm::cl::debug::InvalidDisparityDebugStats stats =
        sgm::cl::debug::compute_invalid_disparity_debug_stats(
            wta_left.data(),
            median_left.data(),
            median_right.data(),
            checked_left.data(),
            left_pixels.data(),
            width,
            height,
            pitch,
            pitch,
            false,
            1);

    require(stats.total == 8, "total pixel count should match width * height");
    require(stats.wta_invalid == 1, "wta invalid count should be classified");
    require(stats.median_invalid == 2, "median invalid count should be classified");
    require(stats.median_added_invalid == 1, "median-added invalid should be classified");
    require(stats.median_removed_invalid == 0, "median-removed invalid should be classified");
    require(stats.final_invalid == 5, "final invalid count should be classified");
    require(stats.mask_zero_new_invalid == 1, "mask-zero invalid should be classified");
    require(stats.lr_right_invalid == 1, "right-invalid LR failure should be classified");
    require(stats.lr_diff_invalid == 1, "LR-diff failure should be classified");
    require(stats.unclassified_new_invalid == 0, "all new invalid pixels should be classified");
}

void test_classifies_subpixel_lr_diff_using_integer_disparity()
{
    const int width = 3;
    const int height = 1;
    const int pitch = 3;

    const std::vector<uint16_t> wta_left = { 16, 32, 16 };
    const std::vector<uint16_t> median_left = { 16, 32, 16 };
    const std::vector<uint16_t> median_right = { 0, 4, 1 };
    const std::vector<uint16_t> checked_left = { 16, 32, sgm::cl::debug::internal_invalid_disparity() };
    const std::vector<uint8_t> left_pixels = { 8, 8, 8 };

    const sgm::cl::debug::InvalidDisparityDebugStats stats =
        sgm::cl::debug::compute_invalid_disparity_debug_stats(
            wta_left.data(),
            median_left.data(),
            median_right.data(),
            checked_left.data(),
            left_pixels.data(),
            width,
            height,
            pitch,
            pitch,
            true,
            1);

    require(stats.lr_diff_invalid == 1, "subpixel left disparity should be shifted before LR check classification");
    require(stats.unclassified_new_invalid == 0, "subpixel LR invalid should be classified");
}

} // namespace

int main()
{
    try
    {
        test_classifies_invalid_sources();
        test_classifies_subpixel_lr_diff_using_integer_disparity();
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_invalid_disparity_debug failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
