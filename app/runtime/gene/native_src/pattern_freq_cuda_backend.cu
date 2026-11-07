#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int kBaseStates = 4;
constexpr int kPatternDim = 256;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

__device__ __forceinline__ unsigned char unpack_low(uint8_t byte) {
    return static_cast<unsigned char>(byte & 0x0F);
}

__device__ __forceinline__ unsigned char unpack_high(uint8_t byte) {
    return static_cast<unsigned char>((byte >> 4) & 0x0F);
}
}  // namespace

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ double warp_sum(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ int warp_sum_int(int value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ int64_t warp_sum_int64(int64_t value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void complete_state_projection_kernel(
    const uint8_t* __restrict__ states,
    const int8_t* __restrict__ directions,
    float* __restrict__ projected,
    int node_count,
    int site_count,
    int projection_count
) {
    const int output = blockIdx.x;
    const int node = output / projection_count;
    const int projection = output - node * projection_count;
    if (node >= node_count) {
        return;
    }
    const uint8_t* row = states + static_cast<int64_t>(node) * site_count;
    int local = 0;
    for (int site = threadIdx.x; site < site_count; site += blockDim.x) {
        const int state = static_cast<int>(row[site]);
        local += static_cast<int>(
            directions[(static_cast<int64_t>(site) * kBaseStates + state) *
                       projection_count + projection]
        );
    }
    local = warp_sum_int(local);
    __shared__ int warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        int total = lane < kWarpsPerBlock ? warp_totals[lane] : 0;
        total = warp_sum_int(total);
        if (lane == 0) {
            projected[output] = static_cast<float>(total);
        }
    }
}

template <typename output_t>
__global__ void complete_state_row_sum_kernel(
    const uint8_t* __restrict__ states,
    const int64_t* __restrict__ site_counts,
    output_t* __restrict__ row_sums,
    int node_count,
    int site_count
) {
    const int node = blockIdx.x;
    if (node >= node_count) {
        return;
    }
    const uint8_t* row = states + static_cast<int64_t>(node) * site_count;
    int64_t local = 0;
    for (int site = threadIdx.x; site < site_count; site += blockDim.x) {
        const int state = static_cast<int>(row[site]);
        local += static_cast<int64_t>(node_count) -
            site_counts[static_cast<int64_t>(site) * kBaseStates + state];
    }
    local = warp_sum_int64(local);
    __shared__ int64_t warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        int64_t total = lane < kWarpsPerBlock ? warp_totals[lane] : int64_t{0};
        total = warp_sum_int64(total);
        if (lane == 0) {
            row_sums[node] = static_cast<output_t>(total) /
                static_cast<output_t>(site_count);
        }
    }
}

__global__ void sparse_state_distance_kernel(
    const uint8_t* __restrict__ states,
    const int64_t* __restrict__ pair_indices,
    float* __restrict__ distances,
    int pair_count,
    int node_count,
    int site_count
) {
    const int pair = blockIdx.x;
    if (pair >= pair_count) {
        return;
    }
    const int64_t left = pair_indices[static_cast<int64_t>(pair) * 2];
    const int64_t right = pair_indices[static_cast<int64_t>(pair) * 2 + 1];
    if (left < 0 || left >= node_count || right < 0 || right >= node_count) {
        return;
    }
    const uint8_t* left_values = states + left * site_count;
    const uint8_t* right_values = states + right * site_count;
    int matches = 0;
    for (int site = threadIdx.x; site < site_count; site += blockDim.x) {
        matches += static_cast<int>(left_values[site] == right_values[site]);
    }
    matches = warp_sum_int(matches);
    __shared__ int warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = matches;
    }
    __syncthreads();
    if (warp == 0) {
        int total = lane < kWarpsPerBlock ? warp_totals[lane] : 0;
        total = warp_sum_int(total);
        if (lane == 0) {
            distances[pair] =
                1.0f - static_cast<float>(total) / static_cast<float>(site_count);
        }
    }
}

__global__ void sparse_profile_distance_kernel(
    const float* __restrict__ profiles,
    const int64_t* __restrict__ pair_indices,
    float* __restrict__ distances,
    int pair_count,
    int node_count,
    int profile_width,
    int site_count
) {
    const int pair = blockIdx.x;
    if (pair >= pair_count) {
        return;
    }
    const int64_t left = pair_indices[static_cast<int64_t>(pair) * 2];
    const int64_t right = pair_indices[static_cast<int64_t>(pair) * 2 + 1];
    if (left < 0 || left >= node_count || right < 0 || right >= node_count) {
        return;
    }
    const float* left_values = profiles + left * profile_width;
    const float* right_values = profiles + right * profile_width;
    float local = 0.0f;
    for (int coordinate = threadIdx.x; coordinate < profile_width;
         coordinate += blockDim.x) {
        local += left_values[coordinate] * right_values[coordinate];
    }
    local = warp_sum(local);
    __shared__ float warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < kWarpsPerBlock ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            distances[pair] = 1.0f - total / static_cast<float>(site_count);
        }
    }
}

__global__ void sparse_marginalized_profile_distance_kernel(
    const float* __restrict__ profiles,
    const float* __restrict__ site_mismatch_prior,
    const int64_t* __restrict__ pair_indices,
    float* __restrict__ distances,
    int pair_count,
    int node_count,
    int site_count
) {
    const int pair = blockIdx.x;
    if (pair >= pair_count) {
        return;
    }
    const int64_t left = pair_indices[static_cast<int64_t>(pair) * 2];
    const int64_t right = pair_indices[static_cast<int64_t>(pair) * 2 + 1];
    if (left < 0 || left >= node_count || right < 0 || right >= node_count) {
        return;
    }
    const int profile_width = site_count * kBaseStates;
    const float* left_values = profiles + left * profile_width;
    const float* right_values = profiles + right * profile_width;
    float local = 0.0f;
    for (int site = threadIdx.x; site < site_count; site += blockDim.x) {
        const int offset = site * kBaseStates;
        float left_mass = 0.0f;
        float right_mass = 0.0f;
        float match = 0.0f;
#pragma unroll
        for (int base = 0; base < kBaseStates; ++base) {
            const float left_value = left_values[offset + base];
            const float right_value = right_values[offset + base];
            left_mass += left_value;
            right_mass += right_value;
            match += left_value * right_value;
        }
        const float prior = site_mismatch_prior[site];
        local += prior + (1.0f - prior) * left_mass * right_mass - match;
    }
    local = warp_sum(local);
    __shared__ float warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < kWarpsPerBlock ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            distances[pair] = total / static_cast<float>(site_count);
        }
    }
}

__global__ void marginalized_profile_row_sum_kernel(
    const float* __restrict__ profiles,
    const double* __restrict__ site_mismatch_prior,
    const double* __restrict__ aggregate_profile,
    double* __restrict__ row_sums,
    int node_count,
    int site_count
) {
    const int node = blockIdx.x;
    if (node >= node_count) {
        return;
    }
    const int profile_width = site_count * kBaseStates;
    const float* values = profiles + static_cast<int64_t>(node) * profile_width;
    double local = 0.0;
    for (int site = threadIdx.x; site < site_count; site += blockDim.x) {
        const int offset = site * kBaseStates;
        double mass = 0.0;
        double aggregate_mass = 0.0;
        double match_all = 0.0;
        double match_self = 0.0;
#pragma unroll
        for (int base = 0; base < kBaseStates; ++base) {
            const double value = static_cast<double>(values[offset + base]);
            const double aggregate = aggregate_profile[offset + base];
            mass += value;
            aggregate_mass += aggregate;
            match_all += value * aggregate;
            match_self += value * value;
        }
        const double prior = site_mismatch_prior[site];
        local += static_cast<double>(node_count - 1) * prior
            + (1.0 - prior) * (mass * aggregate_mass - mass * mass)
            - (match_all - match_self);
    }
    local = warp_sum(local);
    __shared__ double warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        double total = lane < kWarpsPerBlock ? warp_totals[lane] : 0.0;
        total = warp_sum(total);
        if (lane == 0) {
            row_sums[node] = total / static_cast<double>(site_count);
        }
    }
}

__global__ void sparse_coverage_profile_distance_kernel(
    const float* __restrict__ profiles,
    const float* __restrict__ site_mismatch_prior,
    const float* __restrict__ site_frequency,
    const float* __restrict__ site_imputation_weight,
    const int64_t* __restrict__ pair_indices,
    float* __restrict__ distances,
    int pair_count,
    int node_count,
    int site_count
) {
    const int pair = blockIdx.x;
    if (pair >= pair_count) return;
    const int64_t left = pair_indices[static_cast<int64_t>(pair) * 2];
    const int64_t right = pair_indices[static_cast<int64_t>(pair) * 2 + 1];
    if (left < 0 || left >= node_count || right < 0 || right >= node_count) return;
    const int width = site_count * kBaseStates;
    const float* left_values = profiles + left * width;
    const float* right_values = profiles + right * width;
    float local = 0.0f;
    for (int site = threadIdx.x; site < site_count; site += blockDim.x) {
        const int offset = site * kBaseStates;
        const float prior = site_mismatch_prior[site];
        float lm = 0.0f, rm = 0.0f, match = 0.0f, lh = 0.0f, rh = 0.0f;
#pragma unroll
        for (int base = 0; base < kBaseStates; ++base) {
            const float lv = left_values[offset + base];
            const float rv = right_values[offset + base];
            const float residual = 1.0f - site_frequency[offset + base] - prior;
            lm += lv; rm += rv; match += lv * rv;
            lh += lv * residual; rh += rv * residual;
        }
        local += prior + (1.0f - prior) * lm * rm - match
            + site_imputation_weight[site] *
              (lh * (1.0f - rm) + rh * (1.0f - lm));
    }
    local = warp_sum(local);
    __shared__ float warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_totals[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float total = lane < kWarpsPerBlock ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) distances[pair] = total / static_cast<float>(site_count);
    }
}

__global__ void coverage_profile_row_sum_kernel(
    const float* __restrict__ profiles,
    const double* __restrict__ site_mismatch_prior,
    const double* __restrict__ site_frequency,
    const double* __restrict__ site_imputation_weight,
    const double* __restrict__ aggregate_profile,
    double* __restrict__ row_sums,
    int node_count,
    int site_count
) {
    const int node = blockIdx.x;
    if (node >= node_count) return;
    const int width = site_count * kBaseStates;
    const float* values = profiles + static_cast<int64_t>(node) * width;
    double local = 0.0;
    for (int site = threadIdx.x; site < site_count; site += blockDim.x) {
        const int offset = site * kBaseStates;
        const double prior = site_mismatch_prior[site];
        double mass = 0.0, aggregate_mass = 0.0;
        double match_all = 0.0, match_self = 0.0, h = 0.0, aggregate_h = 0.0;
#pragma unroll
        for (int base = 0; base < kBaseStates; ++base) {
            const double value = static_cast<double>(values[offset + base]);
            const double aggregate = aggregate_profile[offset + base];
            const double residual = 1.0 - site_frequency[offset + base] - prior;
            mass += value; aggregate_mass += aggregate;
            match_all += value * aggregate; match_self += value * value;
            h += value * residual; aggregate_h += aggregate * residual;
        }
        local += static_cast<double>(node_count - 1) * prior
            + (1.0 - prior) * (mass * aggregate_mass - mass * mass)
            - (match_all - match_self)
            + site_imputation_weight[site] * (
                h * (static_cast<double>(node_count) - aggregate_mass)
                + (1.0 - mass) * aggregate_h
                - 2.0 * h * (1.0 - mass)
            );
    }
    local = warp_sum(local);
    __shared__ double warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_totals[warp] = local;
    __syncthreads();
    if (warp == 0) {
        double total = lane < kWarpsPerBlock ? warp_totals[lane] : 0.0;
        total = warp_sum(total);
        if (lane == 0) row_sums[node] = total / static_cast<double>(site_count);
    }
}

__global__ void accumulate_sparse_profile_matches_kernel(
    const float* __restrict__ profiles,
    const int64_t* __restrict__ pair_indices,
    double* __restrict__ matches,
    int pair_count,
    int node_count,
    int profile_width
) {
    const int pair = blockIdx.x;
    if (pair >= pair_count) {
        return;
    }
    const int64_t left = pair_indices[static_cast<int64_t>(pair) * 2];
    const int64_t right = pair_indices[static_cast<int64_t>(pair) * 2 + 1];
    if (left < 0 || left >= node_count || right < 0 || right >= node_count) {
        return;
    }
    const float* left_values = profiles + left * profile_width;
    const float* right_values = profiles + right * profile_width;
    double local = 0.0;
    for (int coordinate = threadIdx.x; coordinate < profile_width;
         coordinate += blockDim.x) {
        local += static_cast<double>(left_values[coordinate]) *
                 static_cast<double>(right_values[coordinate]);
    }
    local = warp_sum(local);
    __shared__ double warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        double total = lane < kWarpsPerBlock ? warp_totals[lane] : 0.0;
        total = warp_sum(total);
        if (lane == 0) {
            matches[pair] += total;
        }
    }
}

__global__ void sparse_imputed_state_distance_kernel(
    const uint8_t* __restrict__ states,
    const float* __restrict__ site_frequency,
    const int64_t* __restrict__ pair_indices,
    float* __restrict__ distances,
    int pair_count,
    int node_count,
    int site_count
) {
    const int pair = blockIdx.x;
    if (pair >= pair_count) {
        return;
    }
    const int64_t left = pair_indices[static_cast<int64_t>(pair) * 2];
    const int64_t right = pair_indices[static_cast<int64_t>(pair) * 2 + 1];
    if (left < 0 || left >= node_count || right < 0 || right >= node_count) {
        return;
    }
    const uint8_t* left_values = states + left * site_count;
    const uint8_t* right_values = states + right * site_count;
    const int profile_width = site_count * kBaseStates;
    float local = 0.0f;
    // Preserve the exact flattened coordinate traversal used by
    // sparse_profile_distance_kernel.  Only the storage representation differs:
    // observed cells are implicit one-hot vectors and missing cells read the
    // same float32 empirical site frequency used by the expanded slab.
    for (int coordinate = threadIdx.x; coordinate < profile_width;
         coordinate += blockDim.x) {
        const int site = coordinate / kBaseStates;
        const int base = coordinate - site * kBaseStates;
        const int left_state = static_cast<int>(left_values[site]);
        const int right_state = static_cast<int>(right_values[site]);
        const float left_profile = left_state < kBaseStates
            ? static_cast<float>(left_state == base)
            : site_frequency[coordinate];
        const float right_profile = right_state < kBaseStates
            ? static_cast<float>(right_state == base)
            : site_frequency[coordinate];
        local += left_profile * right_profile;
    }
    local = warp_sum(local);
    __shared__ float warp_totals[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_totals[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < kWarpsPerBlock ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            distances[pair] = 1.0f - total / static_cast<float>(site_count);
        }
    }
}

template <typename output_t>
__global__ void imputed_state_row_sum_kernel(
    const uint8_t* __restrict__ states,
    const float* __restrict__ site_frequency,
    const output_t* __restrict__ aggregate_profile,
    output_t* __restrict__ row_sums,
    int node_count,
    int site_count
) {
    const int node = blockIdx.x;
    if (node >= node_count) {
        return;
    }
    const uint8_t* row = states + static_cast<int64_t>(node) * site_count;
    const int profile_width = site_count * kBaseStates;
    output_t local_all = static_cast<output_t>(0.0);
    output_t local_self = static_cast<output_t>(0.0);
    for (int coordinate = threadIdx.x; coordinate < profile_width;
         coordinate += blockDim.x) {
        const int site = coordinate / kBaseStates;
        const int base = coordinate - site * kBaseStates;
        const int state = static_cast<int>(row[site]);
        const output_t profile = state < kBaseStates
            ? static_cast<output_t>(state == base)
            : static_cast<output_t>(site_frequency[coordinate]);
        local_all += profile * aggregate_profile[coordinate];
        local_self += profile * profile;
    }
    local_all = warp_sum(local_all);
    local_self = warp_sum(local_self);
    __shared__ output_t warp_all[kWarpsPerBlock];
    __shared__ output_t warp_self[kWarpsPerBlock];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_all[warp] = local_all;
        warp_self[warp] = local_self;
    }
    __syncthreads();
    if (warp == 0) {
        output_t total_all = lane < kWarpsPerBlock
            ? warp_all[lane] : static_cast<output_t>(0.0);
        output_t total_self = lane < kWarpsPerBlock
            ? warp_self[lane] : static_cast<output_t>(0.0);
        total_all = warp_sum(total_all);
        total_self = warp_sum(total_self);
        if (lane == 0) {
            const output_t sites = static_cast<output_t>(site_count);
            row_sums[node] =
                (static_cast<output_t>(node_count) -
                 static_cast<output_t>(total_all) / sites) -
                (static_cast<output_t>(1.0) -
                 static_cast<output_t>(total_self) / sites);
        }
    }
}

__global__ void pattern_freq_packed_kernel(
    const uint8_t* __restrict__ sequences_packed,
    const int64_t* __restrict__ quartet_indices,
    float* __restrict__ result,
    int n_quartets,
    int seq_length,
    int packed_seq_length,
    int n_species
) {
    const int quartet_idx = blockIdx.x;
    if (quartet_idx >= n_quartets) {
        return;
    }

    const int sp0 = static_cast<int>(quartet_indices[quartet_idx * 4 + 0]);
    const int sp1 = static_cast<int>(quartet_indices[quartet_idx * 4 + 1]);
    const int sp2 = static_cast<int>(quartet_indices[quartet_idx * 4 + 2]);
    const int sp3 = static_cast<int>(quartet_indices[quartet_idx * 4 + 3]);
    if (sp0 < 0 || sp0 >= n_species || sp1 < 0 || sp1 >= n_species ||
        sp2 < 0 || sp2 >= n_species || sp3 < 0 || sp3 >= n_species) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ int warp_hist[kWarpsPerBlock][kPatternDim];
    __shared__ int valid_count_shared[kThreadsPerBlock];
    __shared__ int total_valid_shared;

    for (int bin = lane; bin < kPatternDim; bin += 32) {
        warp_hist[warp_id][bin] = 0;
    }
    __syncthreads();

    int valid_count = 0;
    const uint8_t* row0 = sequences_packed + static_cast<int64_t>(sp0) * packed_seq_length;
    const uint8_t* row1 = sequences_packed + static_cast<int64_t>(sp1) * packed_seq_length;
    const uint8_t* row2 = sequences_packed + static_cast<int64_t>(sp2) * packed_seq_length;
    const uint8_t* row3 = sequences_packed + static_cast<int64_t>(sp3) * packed_seq_length;

    for (int packed_pos = tid; packed_pos < packed_seq_length; packed_pos += blockDim.x) {
        const uint8_t b0 = row0[packed_pos];
        const uint8_t b1 = row1[packed_pos];
        const uint8_t b2 = row2[packed_pos];
        const uint8_t b3 = row3[packed_pos];

        const int site0 = packed_pos << 1;

        const unsigned char s0_lo = unpack_low(b0);
        const unsigned char s1_lo = unpack_low(b1);
        const unsigned char s2_lo = unpack_low(b2);
        const unsigned char s3_lo = unpack_low(b3);
        if (s0_lo < kBaseStates && s1_lo < kBaseStates && s2_lo < kBaseStates && s3_lo < kBaseStates) {
            const int pattern_idx =
                static_cast<int>(s0_lo) * 64 +
                static_cast<int>(s1_lo) * 16 +
                static_cast<int>(s2_lo) * 4 +
                static_cast<int>(s3_lo);
            atomicAdd(&warp_hist[warp_id][pattern_idx], 1);
            ++valid_count;
        }

        if (site0 + 1 < seq_length) {
            const unsigned char s0_hi = unpack_high(b0);
            const unsigned char s1_hi = unpack_high(b1);
            const unsigned char s2_hi = unpack_high(b2);
            const unsigned char s3_hi = unpack_high(b3);
            if (s0_hi < kBaseStates && s1_hi < kBaseStates && s2_hi < kBaseStates && s3_hi < kBaseStates) {
                const int pattern_idx =
                    static_cast<int>(s0_hi) * 64 +
                    static_cast<int>(s1_hi) * 16 +
                    static_cast<int>(s2_hi) * 4 +
                    static_cast<int>(s3_hi);
                atomicAdd(&warp_hist[warp_id][pattern_idx], 1);
                ++valid_count;
            }
        }
    }

    valid_count_shared[tid] = valid_count;
    __syncthreads();

    if (tid == 0) {
        int total_valid = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            total_valid += valid_count_shared[i];
        }
        total_valid_shared = total_valid;
    }
    __syncthreads();

    float* result_base = result + static_cast<int64_t>(quartet_idx) * kPatternDim;
    const int total_valid = total_valid_shared;
    int count = 0;
    for (int warp = 0; warp < kWarpsPerBlock; ++warp) {
        count += warp_hist[warp][tid];
    }
    if (total_valid > 0) {
        result_base[tid] = static_cast<float>(count) / static_cast<float>(total_valid);
    } else {
        result_base[tid] = 0.0f;
    }
}

__global__ void pattern_freq_packed_blocks_kernel(
    const uint8_t* __restrict__ sequences_packed,
    const int64_t* __restrict__ quartet_indices,
    float* __restrict__ result,
    int n_quartets,
    int seq_length,
    int packed_seq_length,
    int n_species,
    int block_count
) {
    const int output_idx = blockIdx.x;
    const int block_idx = output_idx / n_quartets;
    const int quartet_idx = output_idx - block_idx * n_quartets;
    if (quartet_idx >= n_quartets || block_idx >= block_count) {
        return;
    }

    const int sp0 = static_cast<int>(quartet_indices[quartet_idx * 4 + 0]);
    const int sp1 = static_cast<int>(quartet_indices[quartet_idx * 4 + 1]);
    const int sp2 = static_cast<int>(quartet_indices[quartet_idx * 4 + 2]);
    const int sp3 = static_cast<int>(quartet_indices[quartet_idx * 4 + 3]);
    if (sp0 < 0 || sp0 >= n_species || sp1 < 0 || sp1 >= n_species ||
        sp2 < 0 || sp2 >= n_species || sp3 < 0 || sp3 >= n_species) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    __shared__ int warp_hist[kWarpsPerBlock][kPatternDim];
    __shared__ int valid_count_shared[kThreadsPerBlock];
    __shared__ int total_valid_shared;
    for (int bin = lane; bin < kPatternDim; bin += 32) {
        warp_hist[warp_id][bin] = 0;
    }
    __syncthreads();

    const uint8_t* row0 = sequences_packed + static_cast<int64_t>(sp0) * packed_seq_length;
    const uint8_t* row1 = sequences_packed + static_cast<int64_t>(sp1) * packed_seq_length;
    const uint8_t* row2 = sequences_packed + static_cast<int64_t>(sp2) * packed_seq_length;
    const uint8_t* row3 = sequences_packed + static_cast<int64_t>(sp3) * packed_seq_length;
    const int site_begin = static_cast<int>(
        (static_cast<int64_t>(block_idx) * seq_length) / block_count
    );
    const int site_end = static_cast<int>(
        (static_cast<int64_t>(block_idx + 1) * seq_length) / block_count
    );
    int valid_count = 0;
    for (int site = site_begin + tid; site < site_end; site += blockDim.x) {
        const int packed_pos = site >> 1;
        const bool high = (site & 1) != 0;
        const uint8_t b0 = row0[packed_pos];
        const uint8_t b1 = row1[packed_pos];
        const uint8_t b2 = row2[packed_pos];
        const uint8_t b3 = row3[packed_pos];
        const unsigned char s0 = high ? unpack_high(b0) : unpack_low(b0);
        const unsigned char s1 = high ? unpack_high(b1) : unpack_low(b1);
        const unsigned char s2 = high ? unpack_high(b2) : unpack_low(b2);
        const unsigned char s3 = high ? unpack_high(b3) : unpack_low(b3);
        if (s0 < kBaseStates && s1 < kBaseStates &&
            s2 < kBaseStates && s3 < kBaseStates) {
            const int pattern_idx =
                static_cast<int>(s0) * 64 + static_cast<int>(s1) * 16 +
                static_cast<int>(s2) * 4 + static_cast<int>(s3);
            atomicAdd(&warp_hist[warp_id][pattern_idx], 1);
            ++valid_count;
        }
    }
    valid_count_shared[tid] = valid_count;
    __syncthreads();
    if (tid == 0) {
        int total_valid = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            total_valid += valid_count_shared[i];
        }
        total_valid_shared = total_valid;
    }
    __syncthreads();

    float* result_base = result + static_cast<int64_t>(output_idx) * kPatternDim;
    int count = 0;
    for (int warp = 0; warp < kWarpsPerBlock; ++warp) {
        count += warp_hist[warp][tid];
    }
    if (total_valid_shared > 0) {
        result_base[tid] =
            static_cast<float>(count) / static_cast<float>(total_valid_shared);
    } else {
        result_base[tid] = 0.0f;
    }
}

torch::Tensor compute_pattern_frequencies_cuda_backend_kernel(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length
) {
    c10::cuda::CUDAGuard device_guard(sequences_packed.device());
    const int n_species = static_cast<int>(sequences_packed.size(0));
    const int packed_seq_length = static_cast<int>(sequences_packed.size(1));
    const int n_quartets = static_cast<int>(quartet_indices.size(0));

    auto result = torch::zeros(
        {n_quartets, kPatternDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(sequences_packed.device())
    );

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pattern_freq_packed_kernel<<<n_quartets, kThreadsPerBlock, 0, stream>>>(
        sequences_packed.data_ptr<uint8_t>(),
        quartet_indices.data_ptr<int64_t>(),
        result.data_ptr<float>(),
        n_quartets,
        static_cast<int>(seq_length),
        packed_seq_length,
        n_species
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

torch::Tensor compute_pattern_frequencies_cuda_packed_blocks_backend_kernel(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length,
    int64_t block_count
) {
    c10::cuda::CUDAGuard device_guard(sequences_packed.device());
    const int n_species = static_cast<int>(sequences_packed.size(0));
    const int packed_seq_length = static_cast<int>(sequences_packed.size(1));
    const int n_quartets = static_cast<int>(quartet_indices.size(0));
    const int blocks = static_cast<int>(block_count);
    auto result = torch::zeros(
        {blocks, n_quartets, kPatternDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(sequences_packed.device())
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pattern_freq_packed_blocks_kernel<<<n_quartets * blocks, kThreadsPerBlock, 0, stream>>>(
        sequences_packed.data_ptr<uint8_t>(),
        quartet_indices.data_ptr<int64_t>(),
        result.data_ptr<float>(),
        n_quartets,
        static_cast<int>(seq_length),
        packed_seq_length,
        n_species,
        blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor sparse_profile_distances_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor pair_indices
) {
    c10::cuda::CUDAGuard device_guard(profiles.device());
    const int node_count = static_cast<int>(profiles.size(0));
    const int site_count = static_cast<int>(profiles.size(1));
    const int profile_width = site_count * static_cast<int>(profiles.size(2));
    const int pair_count = static_cast<int>(pair_indices.size(0));
    auto distances = torch::zeros(
        {pair_count},
        torch::TensorOptions().dtype(torch::kFloat32).device(profiles.device())
    );
    if (pair_count == 0) {
        return distances;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparse_profile_distance_kernel<<<pair_count, kThreadsPerBlock, 0, stream>>>(
        profiles.data_ptr<float>(),
        pair_indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        pair_count,
        node_count,
        profile_width,
        site_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return distances;
}

torch::Tensor sparse_marginalized_profile_distances_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor pair_indices
) {
    c10::cuda::CUDAGuard device_guard(profiles.device());
    const int node_count = static_cast<int>(profiles.size(0));
    const int site_count = static_cast<int>(profiles.size(1));
    const int pair_count = static_cast<int>(pair_indices.size(0));
    auto distances = torch::zeros(
        {pair_count},
        torch::TensorOptions().dtype(torch::kFloat32).device(profiles.device())
    );
    if (pair_count == 0) {
        return distances;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparse_marginalized_profile_distance_kernel<<<
        pair_count, kThreadsPerBlock, 0, stream>>>(
        profiles.data_ptr<float>(),
        site_mismatch_prior.data_ptr<float>(),
        pair_indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        pair_count,
        node_count,
        site_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return distances;
}

torch::Tensor marginalized_profile_row_sums_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor aggregate_profile
) {
    c10::cuda::CUDAGuard device_guard(profiles.device());
    const int node_count = static_cast<int>(profiles.size(0));
    const int site_count = static_cast<int>(profiles.size(1));
    auto row_sums = torch::zeros(
        {node_count},
        torch::TensorOptions().dtype(torch::kFloat64).device(profiles.device())
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    marginalized_profile_row_sum_kernel<<<
        node_count, kThreadsPerBlock, 0, stream>>>(
        profiles.data_ptr<float>(),
        site_mismatch_prior.data_ptr<double>(),
        aggregate_profile.data_ptr<double>(),
        row_sums.data_ptr<double>(),
        node_count,
        site_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return row_sums;
}

torch::Tensor sparse_coverage_profile_distances_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor site_frequency,
    torch::Tensor site_imputation_weight,
    torch::Tensor pair_indices
) {
    c10::cuda::CUDAGuard device_guard(profiles.device());
    const int nodes = static_cast<int>(profiles.size(0));
    const int sites = static_cast<int>(profiles.size(1));
    const int pairs = static_cast<int>(pair_indices.size(0));
    auto distances = torch::zeros(
        {pairs}, torch::TensorOptions().dtype(torch::kFloat32).device(profiles.device())
    );
    if (pairs == 0) return distances;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparse_coverage_profile_distance_kernel<<<pairs, kThreadsPerBlock, 0, stream>>>(
        profiles.data_ptr<float>(), site_mismatch_prior.data_ptr<float>(),
        site_frequency.data_ptr<float>(), site_imputation_weight.data_ptr<float>(),
        pair_indices.data_ptr<int64_t>(), distances.data_ptr<float>(),
        pairs, nodes, sites
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return distances;
}

torch::Tensor coverage_profile_row_sums_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor site_frequency,
    torch::Tensor site_imputation_weight,
    torch::Tensor aggregate_profile
) {
    c10::cuda::CUDAGuard device_guard(profiles.device());
    const int nodes = static_cast<int>(profiles.size(0));
    const int sites = static_cast<int>(profiles.size(1));
    auto rows = torch::zeros(
        {nodes}, torch::TensorOptions().dtype(torch::kFloat64).device(profiles.device())
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    coverage_profile_row_sum_kernel<<<nodes, kThreadsPerBlock, 0, stream>>>(
        profiles.data_ptr<float>(), site_mismatch_prior.data_ptr<double>(),
        site_frequency.data_ptr<double>(), site_imputation_weight.data_ptr<double>(),
        aggregate_profile.data_ptr<double>(), rows.data_ptr<double>(), nodes, sites
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return rows;
}

torch::Tensor sparse_state_distances_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor pair_indices
) {
    c10::cuda::CUDAGuard device_guard(states.device());
    const int node_count = static_cast<int>(states.size(0));
    const int site_count = static_cast<int>(states.size(1));
    const int pair_count = static_cast<int>(pair_indices.size(0));
    auto distances = torch::zeros(
        {pair_count},
        torch::TensorOptions().dtype(torch::kFloat32).device(states.device())
    );
    if (pair_count == 0) {
        return distances;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparse_state_distance_kernel<<<pair_count, kThreadsPerBlock, 0, stream>>>(
        states.data_ptr<uint8_t>(),
        pair_indices.data_ptr<int64_t>(),
        distances.data_ptr<float>(),
        pair_count,
        node_count,
        site_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return distances;
}

torch::Tensor sparse_imputed_state_distances_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor site_frequency,
    torch::Tensor pair_indices
) {
    c10::cuda::CUDAGuard device_guard(states.device());
    const int node_count = static_cast<int>(states.size(0));
    const int site_count = static_cast<int>(states.size(1));
    const int pair_count = static_cast<int>(pair_indices.size(0));
    auto distances = torch::zeros(
        {pair_count},
        torch::TensorOptions().dtype(torch::kFloat32).device(states.device())
    );
    if (pair_count == 0) {
        return distances;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparse_imputed_state_distance_kernel<<<pair_count, kThreadsPerBlock, 0, stream>>>(
        states.data_ptr<uint8_t>(), site_frequency.data_ptr<float>(),
        pair_indices.data_ptr<int64_t>(), distances.data_ptr<float>(),
        pair_count, node_count, site_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return distances;
}

void accumulate_sparse_profile_matches_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor pair_indices,
    torch::Tensor matches
) {
    c10::cuda::CUDAGuard device_guard(profiles.device());
    const int node_count = static_cast<int>(profiles.size(0));
    const int profile_width =
        static_cast<int>(profiles.size(1) * profiles.size(2));
    const int pair_count = static_cast<int>(pair_indices.size(0));
    if (pair_count == 0) {
        return;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    accumulate_sparse_profile_matches_kernel<<<
        pair_count, kThreadsPerBlock, 0, stream>>>(
        profiles.data_ptr<float>(), pair_indices.data_ptr<int64_t>(),
        matches.data_ptr<double>(), pair_count, node_count, profile_width
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor complete_state_projections_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor directions
) {
    c10::cuda::CUDAGuard device_guard(states.device());
    const int node_count = static_cast<int>(states.size(0));
    const int site_count = static_cast<int>(states.size(1));
    const int projection_count = static_cast<int>(directions.size(2));
    auto projected = torch::empty(
        {node_count, projection_count},
        torch::TensorOptions().dtype(torch::kFloat32).device(states.device())
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    complete_state_projection_kernel<<<
        node_count * projection_count, kThreadsPerBlock, 0, stream>>>(
        states.data_ptr<uint8_t>(),
        directions.data_ptr<int8_t>(),
        projected.data_ptr<float>(),
        node_count,
        site_count,
        projection_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return projected;
}

torch::Tensor complete_state_row_sums_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor site_counts,
    bool float64_output
) {
    c10::cuda::CUDAGuard device_guard(states.device());
    const int node_count = static_cast<int>(states.size(0));
    const int site_count = static_cast<int>(states.size(1));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (float64_output) {
        auto result = torch::empty(
            {node_count},
            torch::TensorOptions().dtype(torch::kFloat64).device(states.device())
        );
        complete_state_row_sum_kernel<double><<<node_count, kThreadsPerBlock, 0, stream>>>(
            states.data_ptr<uint8_t>(), site_counts.data_ptr<int64_t>(),
            result.data_ptr<double>(), node_count, site_count
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto result = torch::empty(
        {node_count},
        torch::TensorOptions().dtype(torch::kFloat32).device(states.device())
    );
    complete_state_row_sum_kernel<float><<<node_count, kThreadsPerBlock, 0, stream>>>(
        states.data_ptr<uint8_t>(), site_counts.data_ptr<int64_t>(),
        result.data_ptr<float>(), node_count, site_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor imputed_state_row_sums_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor site_frequency,
    torch::Tensor aggregate_profile,
    bool float64_output
) {
    c10::cuda::CUDAGuard device_guard(states.device());
    const int node_count = static_cast<int>(states.size(0));
    const int site_count = static_cast<int>(states.size(1));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (float64_output) {
        auto result = torch::empty(
            {node_count},
            torch::TensorOptions().dtype(torch::kFloat64).device(states.device())
        );
        imputed_state_row_sum_kernel<double><<<node_count, kThreadsPerBlock, 0, stream>>>(
            states.data_ptr<uint8_t>(), site_frequency.data_ptr<float>(),
            aggregate_profile.data_ptr<double>(), result.data_ptr<double>(),
            node_count, site_count
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto result = torch::empty(
        {node_count},
        torch::TensorOptions().dtype(torch::kFloat32).device(states.device())
    );
    imputed_state_row_sum_kernel<float><<<node_count, kThreadsPerBlock, 0, stream>>>(
        states.data_ptr<uint8_t>(), site_frequency.data_ptr<float>(),
        aggregate_profile.data_ptr<float>(), result.data_ptr<float>(),
        node_count, site_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
