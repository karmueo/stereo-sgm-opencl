# Repository Guidelines

## Project Structure & Module Organization

This repository builds `libsgm_ocl`, a C++14 static library for semi-global stereo matching using OpenCL.

- `libsgm_ocl/` contains the public API headers (`libsgm_ocl.h`, `types.h`).
- `src/` contains C++ implementation files and internal headers.
- `src/ocl/` contains OpenCL kernel sources embedded into the library through `CMakeRC.cmake`.
- `examples/` contains the `stereo_movie` OpenCV/OpenCL example.
- `cmake/` contains package config templates used during install.

There is currently no dedicated `tests/` directory.

## Build, Test, and Development Commands

Use an out-of-tree build directory; `build/` and `b/` are ignored.

```sh
cmake -S . -B build -DBUILD_EXAMPLES=ON -DCL_TARGET_OPENCL_VERSION=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The first command configures the library and optional example. The second compiles `libsgm_ocl` and, when enabled, `stereo_movie`.

```sh
cmake --install build --prefix /tmp/libsgm_ocl-install
```

Use this to verify install rules and generated CMake package files.

Run the example with rectified stereo image streams:

```sh
./build/stereo_movie left/%06d.png right/%06d.png --max_disparity=128 --num_path=4
```

## Coding Style & Naming Conventions

Follow the existing C++ style: C++14, four-space indentation, braces on their own line for namespaces/classes/functions, and snake_case for functions, variables, and file names. Class and struct names use `PascalCase`; template parameters and constants follow local usage. Keep public API changes in `libsgm_ocl/` minimal and document behavior in header comments.

For OpenCL kernels, keep files in `src/ocl/`, use `.cl`, and add new kernels to the `OCL_FILES` list in `CMakeLists.txt` so they are embedded.

## Testing Guidelines

No automated test framework is configured. At minimum, validate changes by configuring and building with `BUILD_EXAMPLES=ON`. For algorithm or kernel changes, run `stereo_movie` on a small rectified stereo pair and compare output visually or against a known result. Prefer adding focused tests if a future test framework is introduced.

## Commit & Pull Request Guidelines

The existing history uses short, direct commit subjects such as `input padding to 4` and `fix bugs`. Keep commits concise and action-oriented, and reference issues when relevant (`fix #6 ...`).

Pull requests should include a short summary, build/test commands run, affected platforms or devices, and notes for OpenCL/OpenCV dependency changes. Include screenshots or sample disparity output when changing algorithm behavior or example rendering.
