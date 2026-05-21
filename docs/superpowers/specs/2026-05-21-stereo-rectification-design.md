# SGM 前双目校正设计

## 目标

在两个示例入口中，在执行 SGM 算法之前增加基于标定文件的双目校正：

- `examples/stereo_movie.cpp`
- `examples/stereo_batch.cpp`

该功能使用新增的 `stereo-camchain.yaml` 标定文件，对原始左右相机图像或左右图片先做双目校正，再计算视差。输出仍然是视差图，不做真实距离深度换算。

## 范围

这是示例程序侧的预处理功能，不修改 `libsgm_ocl` 公共 API，也不修改 OpenCL SGM kernel。

实现会新增一组示例共用代码，用于读取标定文件、生成校正映射并应用到左右图像，然后把这组逻辑接入 `stereo_movie` 和 `stereo_batch`。

## 标定文件格式

校正代码支持当前 `stereo-camchain.yaml` 使用的 Kalibr 风格 YAML：

- `cam0` 和 `cam1`
- `camera_model: pinhole`
- `distortion_model: radtan`
- `intrinsics: [fx, fy, cx, cy]`
- `distortion_coeffs: [k1, k2, p1, p2]`
- `resolution: [width, height]`
- `cam1.T_cn_cnm1`，表示两个相机之间的变换

如果相机模型或畸变模型不受支持，程序直接报错。必需字段缺失或格式错误时也直接报错。

## 架构

新增文件：

- `examples/stereo_rectification.h`
- `examples/stereo_rectification.cpp`

共享模块提供一个小的 `StereoRectifier` 类型，负责：

1. 从磁盘读取标定 YAML。
2. 将 Kalibr 内参和畸变参数转换成 OpenCV 相机矩阵。
3. 将双目外参转换成 `cv::stereoRectify` 使用的旋转和平移。
4. 使用 `cv::initUndistortRectifyMap` 生成左右校正映射。
5. 使用 `cv::remap` 对每一对左右图像执行校正。

`CMakeLists.txt` 会把共享源码同时编译进两个示例可执行文件。

## 运行行为

默认启用校正。

两个示例都增加以下参数：

- `--calib <path>`：标定 YAML 路径，默认值为 `stereo-camchain.yaml`。
- `--no_rectify`：关闭校正，保留原来的输入处理流程。

启用校正时：

1. 在创建 SGM buffer 之前加载标定文件。
2. 每一帧或每一对左右图片都会先检查尺寸是否等于标定分辨率。
3. 如果输入尺寸与标定分辨率不一致，程序报错退出。
4. 原始左右图像先通过 `cv::remap` 做校正。
5. 校正后的图像继续走现有流程：裁剪到 4 的倍数、转灰度、上传到 OpenCL、执行 SGM。

校正使用 `cv::stereoRectify`，参数 `alpha=1`，尽量保留完整视场，即使这会带来更多无效黑边。

使用 `--no_rectify` 时，两个示例保持原行为。

## 错误处理

以下情况需要快速失败，并输出清晰错误信息：

- 标定文件无法打开。
- 必需 YAML 字段缺失或格式错误。
- 相机模型不是 `pinhole`。
- 畸变模型不是 `radtan`。
- `cam0` 和 `cam1` 分辨率不一致。
- 输入图像尺寸不等于标定分辨率。
- OpenCV 校正或 remap 输入无效。

## 验证

构建验证：

```sh
cmake -S . -B build -DBUILD_EXAMPLES=ON -DCL_TARGET_OPENCL_VERSION=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

端到端数据验证：

1. 使用 `stereo_batch` 读取仓库的 `data/` 路径。
2. 将生成的视差 PNG 写入输出目录。
3. 确认至少生成一张视差图。

当前 `data` 是指向 `/home/sjgd/Public/stereo/` 的符号链接。如果验证时该目录没有可匹配的左右图片，仍然执行构建验证，并明确报告无法完成端到端数据验证的原因。

## 非目标

- 不生成真实距离深度图。
- 不把输入图片 resize 到标定分辨率。
- 不为不同输入分辨率缩放相机内参。
- 不把 OpenCV 标定逻辑放进公共库 API。
- 不修改 SGM 算法行为或 OpenCL kernel。
