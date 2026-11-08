from pathlib import Path
import json,subprocess,shlex,hashlib,shutil
root=Path(__file__).resolve().parent
project=root.parents[1]/'subprojects-single/scalable-neural-phylogeny'
build=project/'.local/build/dipper-gpu-web-gcc13-d80df351-20260905'
original=project/'.local/research_sources/DIPPER-gpu-web-d80df351-20260905/bin/dipper'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='cacd6caa20cad2f642f5a4f140617ab95938c996de16e485728863c404407153'
objects=root/'objects';objects.mkdir(exist_ok=True)
source=build/'CMakeFiles/dipper.dir/src/tree_generation.cu.o'
renamed=objects/'tree_generation.original.cu.o'
subprocess.run(['objcopy','--redefine-sym','main=dipper_original_main',str(source),str(renamed)],check=True)
wrapper=root/'timing_main.cpp'
wrapper.write_text(r'''
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
''')
seed=root/'seed_control.c'
seed.write_text(r'''
#define _GNU_SOURCE
#include <time.h>
#include <stdlib.h>
#include <dlfcn.h>
time_t time(time_t *out) {
    const char *value = getenv("DIPPER_SHUFFLE_SEED");
    time_t result;
    if (value) result = (time_t)strtoll(value,0,10);
    else { time_t (*original)(time_t*) = dlsym(RTLD_NEXT,"time"); result=original(0); }
    if(out) *out=result;
    return result;
}
''')
commands=[['gcc','-shared','-fPIC','-O2',str(seed),'-ldl','-o',str(root/'seed_control.so')],
['/usr/bin/g++-13','-O3','-std=c++17','-I/usr/local/cuda-12.5/include','-c',str(wrapper),'-o',str(objects/'timing_main.o')]]
for cmd in commands:subprocess.run(cmd,check=True)
link=shlex.split((build/'CMakeFiles/dipper.dir/link.txt').read_text())
command=[];skip=False
for i,part in enumerate(link):
    if skip:skip=False;continue
    if part.startswith('-Wl,--dependency-file='):continue
    if part=='-o':command += ['-o',str(root/'dipper_timed')];skip=True;continue
    if part=='CMakeFiles/dipper.dir/src/tree_generation.cu.o':command.extend([str(renamed),str(objects/'timing_main.o')]);continue
    if part.startswith('CMakeFiles/'):part=str(build/part)
    command.append(part)
subprocess.run(command,check=True,cwd=root)
provenance={'original_binary':str(original),'original_binary_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'source_commit':'d80df351107f05ea2ac01076709defc4ab9d702e','harness':str(root/'dipper_timed'),'harness_sha256':hashlib.sha256((root/'dipper_timed').read_bytes()).hexdigest(),'changes':['original main symbol renamed in copied ELF object; original algorithm and CUDA object instructions reused','outer wrapper initializes CUDA with EAGER modules and TBB before native file-to-tree timer','fixed time() seed shim for controlled stochastic repeats; real clocks unaffected','one 16-CPU process per GPU; no shared-GPU concurrency'],'objects':{str(p.relative_to(build)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (build/'CMakeFiles/dipper.dir').rglob('*.o')},'compile_commands':commands,'link_command':command,'nvidia_timing_reference':'https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/lazy-loading.html'}
(root/'HARNESS.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({k:v for k,v in provenance.items() if k not in ['objects','compile_commands','link_command']},indent=2))
