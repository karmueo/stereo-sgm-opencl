#include "ocl_profiler.h"

#include <cassert>
#include <string>

int main()
{
    sgm::cl::OclKernelProfiler profiler(true);
    profiler.add_completed_sample_for_test("census_left", 1000000);
    profiler.add_completed_sample_for_test("census_left", 3000000);
    profiler.add_completed_sample_for_test("winner_takes_all", 2000000);

    const std::string report = profiler.format_report();
    assert(report.find("OpenCL kernel profile") != std::string::npos);
    assert(report.find("census_left") != std::string::npos);
    assert(report.find("count=2") != std::string::npos);
    assert(report.find("avg=2.000 ms") != std::string::npos);
    assert(report.find("winner_takes_all") != std::string::npos);
    assert(report.find("total=2.000 ms") != std::string::npos);

    profiler.clear();
    assert(profiler.format_report().empty());

    sgm::cl::OclKernelProfiler disabled(false);
    disabled.add_completed_sample_for_test("ignored", 1000000);
    assert(disabled.format_report().empty());

    return 0;
}
