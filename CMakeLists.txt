cmake_minimum_required(VERSION 3.10)

project(libsgm_ocl)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(BUILD_EXAMPLES "Build sgm ocl example" ON)
set(CL_TARGET_OPENCL_VERSION 120 CACHE STRING "OpenCL target version")
message(STATUS "OpenCL target verison: ${CL_TARGET_OPENCL_VERSION}")

find_package(OpenCL REQUIRED)
include(CTest)

include(CMakeRC.cmake)

set(OCL_FILES
    src/ocl/inttypes.cl
    src/ocl/utility.cl
    src/ocl/census.cl
    src/ocl/path_aggregation_common.cl
    src/ocl/path_aggregation_vertical.cl
    src/ocl/path_aggregation_vertical_mali_128.cl
    src/ocl/path_aggregation_horizontal.cl
    src/ocl/path_aggregation_oblique.cl
    src/ocl/winner_takes_all.cl
    src/ocl/winner_takes_all_mali_128.cl
    src/ocl/check_consistency.cl
    src/ocl/median_filter.cl
    src/ocl/correct_disparity_range.cl
)

cmrc_add_resource_library(rc_ocl_sgm NAMESPACE "ocl_sgm" ${OCL_FILES})

set(LIBSGM_SRCS
    src/census_transform.hpp
    src/path_aggregation.h
    src/census.cpp
    src/libsgm_ocl.cpp
    src/path_aggregation.cpp
    src/path_aggregation_vertical.cpp
    src/sgm.hpp
    src/sgm.cpp
    src/device_buffer.hpp
    src/device_kernel.h
    src/invalid_disparity_debug.h
    src/ocl_profiler.h
    src/ocl_runtime_config.h
    src/winner_takes_all.hpp
    src/winner_takes_all.cpp
    src/sgm_details.h
    src/sgm_details.cpp
)

set(LIBSGM_PUBLIC_HDRS
    libsgm_ocl/libsgm_ocl.h
    libsgm_ocl/types.h
)

add_library(libsgm_ocl STATIC ${LIBSGM_SRCS} ${LIBSGM_PUBLIC_HDRS} ${OCL_FILES})
target_compile_definitions(libsgm_ocl PUBLIC -DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
add_library(libsgm_ocl::libsgm_ocl ALIAS libsgm_ocl)
source_group(ocl FILES ${OCL_FILES})
source_group(libsgm_ocl FILES ${LIBSGM_PUBLIC_HDRS})

# hack cmake bug, workaround found at https://gitlab.kitware.com/cmake/cmake/-/issues/15415#note_333595
# cannot link object library, because it requires that library to be exported
#ugly workaround
#   get_target_property(TMP cmrc-base INTERFACE_INCLUDE_DIRECTORIES)
#   target_include_directories(libsgm_ocl PRIVATE "${TMP}")
#little bit nicer
target_link_libraries(libsgm_ocl PRIVATE $<BUILD_INTERFACE:rc_ocl_sgm>)
#for debugging
find_package(OpenCV REQUIRED CONFIG)
target_link_libraries(libsgm_ocl PRIVATE ${OpenCV_LIBRARIES})

set_target_properties(libsgm_ocl PROPERTIES DEBUG_POSTFIX _d RELWITHDEBINFO_POSTFIX _rd)
target_include_directories(libsgm_ocl
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/sgm-cl>
)
target_link_libraries(libsgm_ocl PUBLIC OpenCL::OpenCL)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gen/${PROJECT_NAME}-config-version.cmake"
  VERSION 0.0.2
  COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets DESTINATION 
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake 
        ${CMAKE_CURRENT_BINARY_DIR}/gen/${PROJECT_NAME}-config-version.cmake
    DESTINATION 
        lib/cmake/${PROJECT_NAME})
    
install(EXPORT ${PROJECT_NAME}-targets NAMESPACE libsgm_ocl:: DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES ${LIBSGM_PUBLIC_HDRS} DESTINATION include/libsgm_ocl/libsgm_ocl)


if (BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED CONFIG)
    add_executable(stereo_movie examples/stereo_movie.cpp examples/stereo_rectification.cpp)
    target_include_directories(stereo_movie PRIVATE examples src)
    target_link_libraries(stereo_movie PRIVATE libsgm_ocl::libsgm_ocl ${OpenCV_LIBRARIES})
    add_executable(stereo_batch examples/stereo_batch.cpp examples/stereo_rectification.cpp)
    target_include_directories(stereo_batch PRIVATE examples src)
    target_link_libraries(stereo_batch PRIVATE libsgm_ocl::libsgm_ocl ${OpenCV_LIBRARIES})
endif()

if (BUILD_TESTING)
    add_executable(test_stereo_rectification
        tests/test_stereo_rectification.cpp
        examples/stereo_rectification.cpp
    )
    target_include_directories(test_stereo_rectification PRIVATE examples)
    target_link_libraries(test_stereo_rectification PRIVATE ${OpenCV_LIBRARIES})
    add_test(NAME test_stereo_rectification COMMAND test_stereo_rectification)

    add_executable(test_stereo_batch_output
        tests/test_stereo_batch_output.cpp
    )
    target_include_directories(test_stereo_batch_output PRIVATE examples)
    target_link_libraries(test_stereo_batch_output PRIVATE ${OpenCV_LIBRARIES})
    add_test(NAME test_stereo_batch_output COMMAND test_stereo_batch_output)

    add_executable(test_stereo_batch_preprocess
        tests/test_stereo_batch_preprocess.cpp
    )
    target_include_directories(test_stereo_batch_preprocess PRIVATE examples)
    target_link_libraries(test_stereo_batch_preprocess PRIVATE ${OpenCV_LIBRARIES})
    add_test(NAME test_stereo_batch_preprocess COMMAND test_stereo_batch_preprocess)

    add_executable(test_ocl_runtime_config
        tests/test_ocl_runtime_config.cpp
    )
    target_compile_definitions(test_ocl_runtime_config PRIVATE -DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
    target_include_directories(test_ocl_runtime_config PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_ocl_runtime_config PRIVATE OpenCL::OpenCL)
    add_test(NAME test_ocl_runtime_config COMMAND test_ocl_runtime_config)

    add_executable(test_invalid_disparity_debug
        tests/test_invalid_disparity_debug.cpp
    )
    target_compile_definitions(test_invalid_disparity_debug PRIVATE -DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
    target_include_directories(test_invalid_disparity_debug PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME test_invalid_disparity_debug COMMAND test_invalid_disparity_debug)

    add_executable(test_ocl_profiler
        tests/test_ocl_profiler.cpp
    )
    target_compile_definitions(test_ocl_profiler PRIVATE -DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
    target_include_directories(test_ocl_profiler PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_ocl_profiler PRIVATE OpenCL::OpenCL)
    add_test(NAME test_ocl_profiler COMMAND test_ocl_profiler)
endif()
