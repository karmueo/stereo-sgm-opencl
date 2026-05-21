# 仓库指南

## 项目结构与模块组织

本仓库构建 `libsgm_ocl`，这是一个使用 OpenCL 实现半全局立体匹配的 C++14 静态库。

- `libsgm_ocl/` 包含公共 API 头文件（`libsgm_ocl.h`、`types.h`）。
- `src/` 包含 C++ 实现文件和内部头文件。
- `src/ocl/` 包含 OpenCL kernel 源文件，这些文件会通过 `CMakeRC.cmake` 嵌入到库中。
- `examples/` 包含 `stereo_movie` OpenCV/OpenCL 示例。
- `cmake/` 包含安装时使用的包配置模板。

目前没有专门的 `tests/` 目录。

## 构建、测试与开发命令

使用源码树外的构建目录；`build/` 和 `b/` 已被忽略。

```sh
cmake -S . -B build -DBUILD_EXAMPLES=ON -DCL_TARGET_OPENCL_VERSION=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

第一条命令配置库和可选示例。第二条命令编译 `libsgm_ocl`，并在启用示例时编译 `stereo_movie`。

```sh
cmake --install build --prefix /tmp/libsgm_ocl-install
```

使用该命令验证安装规则以及生成的 CMake 包文件。

在 RK3576 / Mali-G52（Panfrost）设备上运行 OpenCL 示例时，需要启用 Rusticl 的 Panfrost 设备：

```sh
RUSTICL_ENABLE=panfrost clinfo
```

确认输出中能看到 `Mali-G52 r1 (Panfrost)` GPU。未设置该环境变量时，`clinfo` 可能只显示 `rusticl` 平台但设备数为 0，示例会报 `no GPU OpenCL devices found on platform 0`。通过 VS Code CMake 插件运行时，也需要在 `.vscode/settings.json` 的调试环境中保留：

```json
"environment": [
    {
        "name": "RUSTICL_ENABLE",
        "value": "panfrost"
    }
]
```

使用已校正的双目图像流运行示例：

```sh
./build/stereo_movie left/%06d.png right/%06d.png --max_disparity=128 --num_path=4
```

使用 `data/` 下的原始左右图和 `stereo-camchain.yaml` 校正后批量生成视差图：

```sh
RUSTICL_ENABLE=panfrost ./build/stereo_batch \
  --input_dir=data \
  --output_dir=output/rectified_batch \
  --calib=stereo-camchain.yaml \
  --max_disparity=128 \
  --num_path=4
```

## 编码风格与命名约定

遵循现有 C++ 风格：C++14、四空格缩进，命名空间、类和函数的大括号单独占一行，函数、变量和文件名使用 snake_case。类名和结构体名使用 `PascalCase`；模板参数和常量遵循局部既有用法。尽量减少 `libsgm_ocl/` 中的公共 API 变更，并在头文件注释中记录行为。

OpenCL kernel 文件应放在 `src/ocl/` 中，使用 `.cl` 后缀，并将新增 kernel 添加到 `CMakeLists.txt` 的 `OCL_FILES` 列表中，以便嵌入到库里。

## 测试指南

当前未配置自动化测试框架。最低限度应使用 `BUILD_EXAMPLES=ON` 进行配置和构建来验证变更。对于算法或 kernel 变更，应在一组小型已校正双目图像上运行 `stereo_movie`，并通过目视检查或与已知结果对比来验证输出。如果将来引入测试框架，优先添加聚焦的测试。

## 提交与 Pull Request 指南

现有历史使用简短直接的提交标题，例如 `input padding to 4` 和 `fix bugs`。保持提交简洁、面向动作，并在相关时引用 issue（如 `fix #6 ...`）。

Pull request 应包含简短摘要、已运行的构建/测试命令、受影响的平台或设备，以及 OpenCL/OpenCV 依赖变更说明。修改算法行为或示例渲染时，请包含截图或示例视差输出。

## 在动手修改代码之前，或执行复杂任务之前，先问我问题，确认后再开始
