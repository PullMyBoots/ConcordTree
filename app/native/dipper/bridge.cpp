// Thin in-process adapter around pinned upstream DIPPER GPU placement.
// Sequence observations, JC distance and placement kernels are unchanged.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <numeric>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include "upstream/mash_placement.cuh"

// The public engine only accepts aligned packed DNA. Reject unreachable
// distance-matrix/MASH branches without linking their unused GPU pipelines.
void MashPlacement::MashDeviceArrays::distConstructionOnGpu(Param&, int, double*) const {
    throw std::logic_error("the packed DIPPER adapter requires aligned DNA");
}
void MashPlacement::MatrixReader::distConstructionOnGpu(Param&, int, double*) {
    throw std::logic_error("the packed DIPPER adapter does not accept a distance matrix");
}

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
static void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("DIPPER ") + operation + ": " + cudaGetErrorString(status));
}
struct DeviceMemory {
    void* pointer = nullptr;
    explicit DeviceMemory(size_t bytes) { checked(cudaMalloc(&pointer, bytes), "allocate"); }
    ~DeviceMemory() { if (pointer) cudaFree(pointer); }
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
};

py::dict build(py::array_t<uint8_t, py::array::c_style> packed,
               int64_t sites, uint32_t seed, int threads) {
    if (packed.ndim() != 2 || packed.shape(0) < 4 || sites < 1 || sites > INT_MAX - 16 ||
        packed.shape(0) > INT_MAX / 40 || packed.shape(1) != (sites + 1) / 2 ||
        threads < 1 || threads > 64)
        throw std::invalid_argument("invalid packed DNA shape, site count or CPU threads");
    const int n = packed.shape(0);
    const size_t words = (sites + 15) / 16, stride = packed.shape(1);
    if (words > SIZE_MAX / sizeof(uint64_t) / size_t(n))
        throw std::overflow_error("DIPPER packed sequence size overflow");
    const size_t sequence_bytes = size_t(n) * words * sizeof(uint64_t);
    // One aligned allocation for topology and scratch, avoiding 11 allocations.
    const size_t topology_bytes = size_t(n) * 448;
    py::array_t<int32_t> permutation(n);
    std::vector<int32_t> heads(2 * size_t(n)), ends(8 * size_t(n)), next(8 * size_t(n));
    double preparation, placement, download;
    {
        py::gil_scoped_release release;
        const auto started = Clock::now();
        size_t free_bytes, total_bytes;
        checked(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
        // Include upstream Thrust's linear placement workspace and an error margin.
        if (sequence_bytes > free_bytes || topology_bytes + size_t(n) * 128 > free_bytes - sequence_bytes)
            throw std::runtime_error("insufficient free GPU memory for DIPPER packed sequences and linear workspace");
        std::vector<int32_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(ids.begin(), ids.end(), rng);
        auto* original = permutation.mutable_data();
        for (int i = 0; i < n; ++i) original[ids[i]] = i;
        // Both runtimes encode A/C/G/T as 0/1/2/3 in little-endian nibbles.
        // Unknown states remain >=4. Only row padding and permutation are needed.
        // No serial full-MSA zero fill: parallel first-touch/copy, then clear
        // at most seven padding bytes per row, matching the upstream layout.
        std::unique_ptr<uint64_t[]> input(new uint64_t[size_t(n) * words]);
        std::vector<uint8_t> observed(n, 0);
        const auto* source = packed.data();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int i = 0; i < n; ++i) {
            auto* row = input.get() + size_t(ids[i]) * words;
            std::memcpy(row, source + size_t(i) * stride, stride);
            std::memset(reinterpret_cast<uint8_t*>(row) + stride, 0, words * sizeof(uint64_t) - stride);
            if (sites & 1) reinterpret_cast<uint8_t*>(row)[stride - 1] &= 15;
            // Usually exits on the first byte; never mistake tail padding for data.
            const auto* original_row = source + size_t(i) * stride;
            for (size_t j = 0; j < stride; ++j) {
                if ((original_row[j] & 15) < 4 ||
                    (2 * j + 1 < size_t(sites) && (original_row[j] >> 4) < 4)) {
                    observed[i] = 1; break;
                }
            }
        }
        if (std::find(observed.begin(), observed.end(), 0) != observed.end())
            throw std::invalid_argument("entirely unobserved taxon in DIPPER input");
        DeviceMemory sequences(sequence_bytes), topology(topology_bytes);
        checked(cudaMemcpy(sequences.pointer, input.get(), sequence_bytes, cudaMemcpyHostToDevice), "upload packed input");
        input.reset();
        MashPlacement::MSADeviceArrays msa{};
        msa.numSequences = n; msa.seqLen = int(sites);
        msa.d_compressedSeqs = static_cast<uint64_t*>(sequences.pointer);
        MashPlacement::KPlacementDeviceArrays tree{};
        tree.numSequences = n; tree.bd = 2; tree.idx = 0; tree.backboneSize = -1;
        auto* cursor = static_cast<uint8_t*>(topology.pointer);
        auto take = [&](size_t bytes) { auto* p = cursor; cursor += bytes; return p; };
        // Put all doubles first so every field is naturally aligned.
        tree.d_dist = reinterpret_cast<double*>(take(size_t(n) * 8));
        tree.d_len = reinterpret_cast<double*>(take(size_t(n) * 8 * 8));
        tree.d_closest_dis = reinterpret_cast<double*>(take(size_t(n) * 20 * 8));
        tree.d_dis_iter = reinterpret_cast<double*>(take(size_t(n) * 2 * 8));
        tree.d_head = reinterpret_cast<int*>(take(size_t(n) * 2 * 4));
        tree.d_e = reinterpret_cast<int*>(take(size_t(n) * 8 * 4));
        tree.d_nxt = reinterpret_cast<int*>(take(size_t(n) * 8 * 4));
        tree.d_belong = reinterpret_cast<int*>(take(size_t(n) * 8 * 4));
        tree.d_closest_id = reinterpret_cast<int*>(take(size_t(n) * 20 * 4));
        tree.d_id_iter = reinterpret_cast<int*>(take(size_t(n) * 2 * 4));
        tree.d_from_iter = reinterpret_cast<int*>(take(size_t(n) * 2 * 4));
        if (size_t(cursor - static_cast<uint8_t*>(topology.pointer)) != topology_bytes)
            throw std::logic_error("DIPPER workspace size mismatch");
        MashPlacement::MashDeviceArrays unused_mash{};
        MashPlacement::MatrixReader unused_matrix{};
        MashPlacement::Param params(15, 1000, 1, 2, "m", "t");
        params.range = {0, -1}; params.totalNumSeqs = n; params.backboneSize = 0; params.batchSize = n;
        checked(cudaDeviceSynchronize(), "prepare"); preparation = seconds(started);
        const auto search = Clock::now();
        tree.findPlacementTree(params, unused_mash, unused_matrix, msa);
        checked(cudaGetLastError(), "launch placement kernels");
        checked(cudaDeviceSynchronize(), "finish placement"); placement = seconds(search);
        const auto copy = Clock::now();
        checked(cudaMemcpy(heads.data(), tree.d_head, heads.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download heads");
        checked(cudaMemcpy(ends.data(), tree.d_e, ends.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download edges");
        checked(cudaMemcpy(next.data(), tree.d_nxt, next.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download links");
        download = seconds(copy);
    }
    // Export topology directly; no Newick serialization/parsing in the handoff.
    std::vector<std::pair<int32_t, int32_t>> edges;
    for (int u = 0; u < 2 * n; ++u) {
        int degree = 0;
        for (int e = heads[u]; e != -1; e = next[e]) {
            if (e < 0 || e >= 8 * n || ends[e] < 0 || ends[e] >= 2 * n || ++degree > 3)
                throw std::runtime_error("invalid DIPPER adjacency");
            if (u < ends[e]) edges.emplace_back(u, ends[e]);
        }
        if (u < n && degree != 1) throw std::runtime_error("DIPPER lost a sequence leaf");
    }
    py::array_t<int32_t> links({py::ssize_t(edges.size()), py::ssize_t(2)});
    auto* link_data = links.mutable_data();
    for (size_t i = 0; i < edges.size(); ++i) {
        link_data[2 * i] = edges[i].first; link_data[2 * i + 1] = edges[i].second;
    }
    py::dict result;
    result["edges"] = links; result["original_row_for_leaf"] = permutation;
    result["preparation_seconds"] = preparation; result["placement_seconds"] = placement;
    result["download_seconds"] = download; result["sequence_bytes"] = sequence_bytes;
    result["topology_workspace_bytes"] = topology_bytes;
    return result;
}

PYBIND11_MODULE(_dipper_native, m) {
    m.doc() = "Pinned DIPPER full-site GPU placement on existing packed DNA";
    m.def("build", &build, py::arg("packed").noconvert(), py::arg("sites"), py::arg("seed"), py::arg("threads"));
}
