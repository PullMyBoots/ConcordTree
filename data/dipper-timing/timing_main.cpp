
#include <cuda_runtime.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <atomic>
extern "C" int dipper_original_main(int, char**);
int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 16);
    const bool warm = std::getenv("DIPPER_TIMING_WARM") && std::atoi(std::getenv("DIPPER_TIMING_WARM"));
    auto init = clock::now();
    if (warm) {
        auto err = cudaSetDevice(0);
        if (err == cudaSuccess) err = cudaFree(nullptr);
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { std::fprintf(stderr,"CUDA prewarm failed: %s\n",cudaGetErrorString(err)); return 90; }
        std::atomic<int> touched{0};
        tbb::parallel_for(0, 4096, [&](int) {touched.fetch_add(1,std::memory_order_relaxed);});
    }
    const double init_seconds = std::chrono::duration<double>(clock::now()-init).count();
    auto start = clock::now();
    const int result = dipper_original_main(argc,argv);
    const auto sync = cudaDeviceSynchronize();
    const double seconds = std::chrono::duration<double>(clock::now()-start).count();
    if (auto path = std::getenv("DIPPER_TIMING_JSON")) {
        std::ofstream out(path);
        out << std::setprecision(17) << "{\"warm\":" << (warm?"true":"false")
            << ",\"initialization_seconds\":" << init_seconds
            << ",\"native_file_to_tree_seconds\":" << seconds
            << ",\"returncode\":" << result << ",\"cuda_sync_code\":" << static_cast<int>(sync) << "}\n";
    }
    return result ? result : (sync == cudaSuccess ? 0 : 91);
}
