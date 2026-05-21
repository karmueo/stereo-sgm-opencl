#pragma once

#include "ocl_runtime_config.h"

#include <CL/cl.h>
#include <chrono>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sgm
{
namespace cl
{

class OclKernelProfiler
{
public:
    typedef std::chrono::steady_clock Clock;

    explicit OclKernelProfiler(bool enabled)
        : enabled_(enabled)
    {
    }

    bool enabled() const
    {
        return enabled_;
    }

    bool event_profiling_enabled() const
    {
        return enabled_ && event_profiling_enabled_;
    }

    void set_enabled(bool enabled)
    {
        if (!enabled)
        {
            clear();
        }
        enabled_ = enabled;
    }

    void set_event_profiling_enabled(bool enabled)
    {
        event_profiling_enabled_ = enabled;
    }

    Clock::time_point kernel_start() const
    {
        return Clock::now();
    }

    void complete_kernel(
        const std::string& name,
        cl_command_queue queue,
        cl_event event,
        Clock::time_point start)
    {
        if (!enabled_)
        {
            if (event != nullptr)
            {
                clReleaseEvent(event);
            }
            return;
        }

        if (event_profiling_enabled_ && event != nullptr)
        {
            record_event(name, event);
            return;
        }

        if (event != nullptr)
        {
            clReleaseEvent(event);
        }
        clFinish(queue);
        const Clock::time_point end = Clock::now();
        const cl_ulong duration_ns =
            static_cast<cl_ulong>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        add_sample(name, duration_ns);
    }

    void record_event(const std::string& name, cl_event event)
    {
        if (event == nullptr)
        {
            return;
        }
        if (!enabled_)
        {
            clReleaseEvent(event);
            return;
        }
        PendingEvent pending;
        pending.name = name;
        pending.event = event;
        pending_events_.push_back(pending);
    }

    void collect_pending()
    {
        for (const auto& pending : pending_events_)
        {
            cl_ulong start = 0;
            cl_ulong end = 0;
            cl_int err_start = clGetEventProfilingInfo(
                pending.event,
                CL_PROFILING_COMMAND_START,
                sizeof(start),
                &start,
                nullptr);
            cl_int err_end = clGetEventProfilingInfo(
                pending.event,
                CL_PROFILING_COMMAND_END,
                sizeof(end),
                &end,
                nullptr);
            if (err_start == CL_SUCCESS && err_end == CL_SUCCESS && end >= start)
            {
                add_sample(pending.name, end - start);
            }
            clReleaseEvent(pending.event);
        }
        pending_events_.clear();
    }

    void add_completed_sample_for_test(const std::string& name, cl_ulong duration_ns)
    {
        if (enabled_)
        {
            add_sample(name, duration_ns);
        }
    }

    std::string format_report() const
    {
        if (!enabled_ || samples_.empty())
        {
            return std::string();
        }

        std::ostringstream out;
        out << "OpenCL kernel profile:" << std::endl;
        out << std::fixed << std::setprecision(3);
        for (const auto& item : samples_)
        {
            const double total_ms = static_cast<double>(item.second.total_ns) / 1000000.0;
            const double avg_ms = item.second.count > 0 ? total_ms / item.second.count : 0.0;
            out << "  " << item.first
                << ": count=" << item.second.count
                << ", total=" << total_ms << " ms"
                << ", avg=" << avg_ms << " ms"
                << std::endl;
        }
        return out.str();
    }

    void clear()
    {
        for (const auto& pending : pending_events_)
        {
            clReleaseEvent(pending.event);
        }
        pending_events_.clear();
        samples_.clear();
    }

private:
    struct PendingEvent
    {
        std::string name;
        cl_event event = nullptr;
    };

    struct SampleStats
    {
        unsigned int count = 0;
        cl_ulong total_ns = 0;
    };

    void add_sample(const std::string& name, cl_ulong duration_ns)
    {
        SampleStats& stats = samples_[name];
        ++stats.count;
        stats.total_ns += duration_ns;
    }

    bool enabled_ = false;
    bool event_profiling_enabled_ = true;
    std::vector<PendingEvent> pending_events_;
    std::map<std::string, SampleStats> samples_;
};

inline OclKernelProfiler& global_ocl_profiler()
{
    static OclKernelProfiler profiler(
        ocl_runtime_config::env_enabled("LIBSGM_OCL_PROFILE"));
    return profiler;
}

inline cl_command_queue_properties profiling_queue_properties()
{
    return global_ocl_profiler().enabled() ? CL_QUEUE_PROFILING_ENABLE : 0;
}

inline cl_command_queue create_ocl_command_queue(
    cl_context context,
    cl_device_id device,
    cl_int* error)
{
    const cl_command_queue_properties properties = profiling_queue_properties();
    if (properties != 0)
    {
        cl_platform_id platform = nullptr;
        clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
        typedef cl_command_queue(CL_API_CALL *CreateQueueWithPropertiesFn)(
            cl_context,
            cl_device_id,
            const cl_properties*,
            cl_int*);
        CreateQueueWithPropertiesFn create_queue_with_properties =
            reinterpret_cast<CreateQueueWithPropertiesFn>(
                clGetExtensionFunctionAddressForPlatform(
                    platform,
                    "clCreateCommandQueueWithProperties"));
        if (create_queue_with_properties == nullptr)
        {
            create_queue_with_properties = reinterpret_cast<CreateQueueWithPropertiesFn>(
                clGetExtensionFunctionAddressForPlatform(
                    platform,
                    "clCreateCommandQueueWithPropertiesKHR"));
        }
        if (create_queue_with_properties == nullptr)
        {
        }
        if (create_queue_with_properties != nullptr)
        {
            const cl_properties queue_properties[] = {
                CL_QUEUE_PROPERTIES,
                static_cast<cl_properties>(properties),
                0
            };
            cl_command_queue queue = create_queue_with_properties(
                context,
                device,
                queue_properties,
                error);
            if (queue != nullptr && (error == nullptr || *error == CL_SUCCESS))
            {
                return queue;
            }
        }

        cl_command_queue queue = clCreateCommandQueue(context, device, properties, error);
        if (queue != nullptr && (error == nullptr || *error == CL_SUCCESS))
        {
            return queue;
        }

        global_ocl_profiler().set_event_profiling_enabled(false);
    }

    return clCreateCommandQueue(context, device, 0, error);
}

} // namespace cl
} // namespace sgm
