# Stereo Rectification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rectify raw stereo image pairs with `stereo-camchain.yaml` before running SGM in both example programs, then verify disparity PNG generation from `data/`.

**Architecture:** Keep rectification as example-only shared code in `examples/stereo_rectification.h/.cpp`. Both `stereo_movie` and `stereo_batch` enable rectification by default, with `--calib` for calibration path and `--no_rectify` for the old path. Add a small C++ CTest executable for parser and rectifier behavior, using no external test framework.

**Tech Stack:** C++14, OpenCV calib3d/imgproc/core, OpenCL, CMake/CTest, existing `stereo_batch` end-to-end example.

---

## File Structure

- Create `examples/stereo_rectification.h`: public example-side `StereoRectifier` API and calibration data structs.
- Create `examples/stereo_rectification.cpp`: Kalibr YAML parsing, `cv::stereoRectify`, remap generation, input-size validation, `cv::remap` application.
- Create `tests/test_stereo_rectification.cpp`: standalone C++ test binary that writes temporary calibration YAML and validates parser/rectifier behavior.
- Modify `CMakeLists.txt`: compile shared rectification source into both examples; enable CTest; add `test_stereo_rectification`.
- Modify `examples/stereo_movie.cpp`: parse `--calib` and `--no_rectify`, apply rectification before cropping/grayscale/SGM, size SGM from rectified first frame.
- Modify `examples/stereo_batch.cpp`: parse `--calib` and `--no_rectify`, apply rectification when loading pairs, preserve existing batch output behavior.

## Task 1: Add Failing Rectification Tests

**Files:**
- Create: `tests/test_stereo_rectification.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create the test source**

Create `tests/test_stereo_rectification.cpp` with tests for:

- loading a Kalibr camchain file,
- rejecting an image whose size differs from calibration resolution,
- rectifying a matching-size image pair.

Use this exact structure:

```cpp
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>

#include "stereo_rectification.h"

namespace
{

std::string write_test_calibration()
{
    const std::string path = "/tmp/libsgm_ocl_test_camchain.yaml";
    std::ofstream out(path.c_str());
    out << "cam0:\n"
        << "  cam_overlaps: [1]\n"
        << "  camera_model: pinhole\n"
        << "  distortion_coeffs: [0.01, -0.02, 0.001, -0.001]\n"
        << "  distortion_model: radtan\n"
        << "  intrinsics: [100.0, 101.0, 32.0, 24.0]\n"
        << "  resolution: [64, 48]\n"
        << "  rostopic: /cam0/image_raw\n"
        << "cam1:\n"
        << "  T_cn_cnm1:\n"
        << "  - [1.0, 0.0, 0.0, -0.1]\n"
        << "  - [0.0, 1.0, 0.0, 0.0]\n"
        << "  - [0.0, 0.0, 1.0, 0.0]\n"
        << "  - [0.0, 0.0, 0.0, 1.0]\n"
        << "  cam_overlaps: [0]\n"
        << "  camera_model: pinhole\n"
        << "  distortion_coeffs: [0.011, -0.021, 0.0011, -0.0011]\n"
        << "  distortion_model: radtan\n"
        << "  intrinsics: [102.0, 103.0, 33.0, 25.0]\n"
        << "  resolution: [64, 48]\n"
        << "  rostopic: /cam1/image_raw\n";
    return path;
}

void require(bool value, const std::string& message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}

void test_loads_calibration()
{
    const stereo_examples::StereoRectifier rectifier(write_test_calibration());
    require(rectifier.image_size() == cv::Size(64, 48), "unexpected calibration resolution");
}

void test_rejects_wrong_input_size()
{
    const stereo_examples::StereoRectifier rectifier(write_test_calibration());
    cv::Mat left(40, 64, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::Mat right(40, 64, CV_8UC3, cv::Scalar(4, 5, 6));
    cv::Mat rectified_left;
    cv::Mat rectified_right;

    bool threw = false;
    try
    {
        rectifier.rectify(left, right, rectified_left, rectified_right);
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    require(threw, "wrong input size should throw");
}

void test_rectifies_matching_input_size()
{
    const stereo_examples::StereoRectifier rectifier(write_test_calibration());
    cv::Mat left(48, 64, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat right(48, 64, CV_8UC3, cv::Scalar(40, 50, 60));
    cv::Mat rectified_left;
    cv::Mat rectified_right;

    rectifier.rectify(left, right, rectified_left, rectified_right);
    require(rectified_left.size() == cv::Size(64, 48), "left rectified size mismatch");
    require(rectified_right.size() == cv::Size(64, 48), "right rectified size mismatch");
    require(rectified_left.type() == left.type(), "left rectified type mismatch");
    require(rectified_right.type() == right.type(), "right rectified type mismatch");
}

} // namespace

int main()
{
    try
    {
        test_loads_calibration();
        test_rejects_wrong_input_size();
        test_rectifies_matching_input_size();
    }
    catch (const std::exception& e)
    {
        std::cerr << "test_stereo_rectification failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```

- [ ] **Step 2: Wire the test into CMake**

Add CTest support and a test executable that references the not-yet-existing shared module:

```cmake
include(CTest)
if (BUILD_TESTING)
    add_executable(test_stereo_rectification
        tests/test_stereo_rectification.cpp
        examples/stereo_rectification.cpp
    )
    target_include_directories(test_stereo_rectification PRIVATE examples)
    target_link_libraries(test_stereo_rectification PRIVATE ${OpenCV_LIBRARIES})
    add_test(NAME test_stereo_rectification COMMAND test_stereo_rectification)
endif()
```

- [ ] **Step 3: Run the test build and verify RED**

Run:

```sh
cmake -S . -B build -DBUILD_EXAMPLES=ON -DCL_TARGET_OPENCL_VERSION=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_stereo_rectification
```

Expected: build fails because `examples/stereo_rectification.h` and `examples/stereo_rectification.cpp` do not exist yet.

## Task 2: Implement Shared Rectification Module

**Files:**
- Create: `examples/stereo_rectification.h`
- Create: `examples/stereo_rectification.cpp`
- Test: `tests/test_stereo_rectification.cpp`

- [ ] **Step 1: Add the public header**

Create `examples/stereo_rectification.h`:

```cpp
#ifndef STEREO_RECTIFICATION_H
#define STEREO_RECTIFICATION_H

#include <string>

#include <opencv2/core/core.hpp>

namespace stereo_examples
{

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
```

- [ ] **Step 2: Implement a narrow Kalibr YAML parser and rectifier**

Create `examples/stereo_rectification.cpp`. Requirements:

- parse only fields required by current `stereo-camchain.yaml`,
- do not use OpenCV `FileStorage`, because it rejects this YAML format,
- support `pinhole` and `radtan` only,
- require same `cam0/cam1` resolution,
- call `cv::stereoRectify(..., 1.0, image_size, ...)`,
- use `cv::initUndistortRectifyMap(..., CV_32FC1, ...)`,
- validate input size in `rectify`.

The key helper behavior should match these signatures:

```cpp
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

StereoCalibration load_kalibr_calibration(const std::string& path);

} // namespace
```

- [ ] **Step 3: Run the unit test and verify GREEN**

Run:

```sh
cmake --build build --target test_stereo_rectification
ctest --test-dir build --output-on-failure -R test_stereo_rectification
```

Expected: `test_stereo_rectification` passes.

## Task 3: Integrate Rectification Into Examples

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `examples/stereo_movie.cpp`
- Modify: `examples/stereo_batch.cpp`

- [ ] **Step 1: Compile shared source into both examples**

Update the examples section in `CMakeLists.txt` so both example targets include `examples/stereo_rectification.cpp`:

```cmake
if (BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED CONFIG)
    add_executable(stereo_movie examples/stereo_movie.cpp examples/stereo_rectification.cpp)
    target_include_directories(stereo_movie PRIVATE examples)
    target_link_libraries(stereo_movie PRIVATE libsgm_ocl::libsgm_ocl ${OpenCV_LIBRARIES})
    add_executable(stereo_batch examples/stereo_batch.cpp examples/stereo_rectification.cpp)
    target_include_directories(stereo_batch PRIVATE examples)
    target_link_libraries(stereo_batch PRIVATE libsgm_ocl::libsgm_ocl ${OpenCV_LIBRARIES})
endif()
```

- [ ] **Step 2: Add CLI parameters to `stereo_movie`**

Add to the command-line keys:

```cpp
"{ calib | stereo-camchain.yaml | Stereo calibration YAML path }"
"{ no_rectify | false | Disable stereo rectification }"
```

Include the header:

```cpp
#include "stereo_rectification.h"
```

Create the optional rectifier before SGM is constructed:

```cpp
const bool rectify_enabled = !parser.get<bool>("no_rectify");
std::unique_ptr<stereo_examples::StereoRectifier> rectifier;
if (rectify_enabled)
{
    rectifier.reset(new stereo_examples::StereoRectifier(parser.get<std::string>("calib")));
}
```

Apply rectification to the first frame before determining `img_size`, `width`, and `height`, and apply it inside the frame loop before cropping/grayscale.

- [ ] **Step 3: Add CLI parameters to `stereo_batch`**

Add to the command-line keys:

```cpp
"{ calib | stereo-camchain.yaml | Stereo calibration YAML path }"
"{ no_rectify | false | Disable stereo rectification }"
```

Include the header:

```cpp
#include "stereo_rectification.h"
```

Construct the optional rectifier before loading pairs:

```cpp
std::unique_ptr<stereo_examples::StereoRectifier> rectifier;
if (!parser.get<bool>("no_rectify"))
{
    rectifier.reset(new stereo_examples::StereoRectifier(parser.get<std::string>("calib")));
}
std::vector<LoadedPair> loaded = load_pairs(pairs, rectifier.get());
```

Change `load_pairs` to accept `const stereo_examples::StereoRectifier* rectifier` and apply `rectifier->rectify(left, right, left, right)` before crop-to-4.

- [ ] **Step 4: Build both examples**

Run:

```sh
cmake --build build --target stereo_movie stereo_batch
```

Expected: both targets build successfully.

## Task 4: Verify End-to-End Disparity Output

**Files:**
- Runtime output only: `output/rectified_batch/`

- [ ] **Step 1: Run all configured tests**

Run:

```sh
ctest --test-dir build --output-on-failure
```

Expected: `test_stereo_rectification` passes.

- [ ] **Step 2: Inspect data path for input naming**

Run:

```sh
find data -maxdepth 2 -type f | sort | head -40
```

Expected: identify left/right image prefixes. If no files are listed, report that data verification is blocked by empty `data/`.

- [ ] **Step 3: Run batch disparity generation**

If `data/` contains files matching default prefixes `left_*.png` and `right_*.png`, run:

```sh
./build/stereo_batch --input_dir=data --output_dir=output/rectified_batch --calib=stereo-camchain.yaml --max_disparity=128 --num_path=4
```

If the prefixes differ, pass the matching `--left_prefix` and `--right_prefix`.

Expected: command exits 0 and writes at least one `output/rectified_batch/disparity_*.png`.

- [ ] **Step 4: Confirm output exists**

Run:

```sh
find output/rectified_batch -maxdepth 1 -name 'disparity_*.png' | sort | head
```

Expected: at least one disparity PNG path is printed.

