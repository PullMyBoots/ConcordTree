#include <torch/extension.h>

#include <stdexcept>

torch::Tensor compute_pattern_frequencies_cuda_backend_kernel(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length
);

torch::Tensor compute_pattern_frequencies_cuda_packed_blocks_backend_kernel(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length,
    int64_t block_count
);

torch::Tensor sparse_profile_distances_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor pair_indices
);

torch::Tensor sparse_marginalized_profile_distances_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor pair_indices
);

torch::Tensor marginalized_profile_row_sums_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor aggregate_profile
);

torch::Tensor sparse_coverage_profile_distances_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor site_frequency,
    torch::Tensor site_imputation_weight,
    torch::Tensor pair_indices
);

torch::Tensor coverage_profile_row_sums_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor site_frequency,
    torch::Tensor site_imputation_weight,
    torch::Tensor aggregate_profile
);

torch::Tensor sparse_state_distances_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor pair_indices
);

torch::Tensor sparse_imputed_state_distances_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor site_frequency,
    torch::Tensor pair_indices
);

void accumulate_sparse_profile_matches_cuda_backend_kernel(
    torch::Tensor profiles,
    torch::Tensor pair_indices,
    torch::Tensor matches
);

torch::Tensor complete_state_projections_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor directions
);

torch::Tensor complete_state_row_sums_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor site_counts,
    bool float64_output
);

torch::Tensor imputed_state_row_sums_cuda_backend_kernel(
    torch::Tensor states,
    torch::Tensor site_frequency,
    torch::Tensor aggregate_profile,
    bool float64_output
);

torch::Tensor compute_pattern_frequencies_cuda_packed(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length
) {
    if (!sequences_packed.is_cuda() || !quartet_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (sequences_packed.dtype() != torch::kUInt8) {
        throw std::runtime_error("packed sequences must be torch.uint8");
    }
    if (quartet_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("quartet_indices must be torch.int64");
    }
    if (sequences_packed.dim() != 2) {
        throw std::runtime_error("packed sequences must have shape (n_species, packed_seq_length)");
    }
    if (quartet_indices.dim() != 2 || quartet_indices.size(1) != 4) {
        throw std::runtime_error("quartet_indices must have shape (n_quartets, 4)");
    }
    if (seq_length <= 0) {
        throw std::runtime_error("seq_length must be > 0");
    }

    return compute_pattern_frequencies_cuda_backend_kernel(
        sequences_packed.contiguous(),
        quartet_indices.contiguous(),
        seq_length
    );
}

torch::Tensor compute_pattern_frequencies_cuda_packed_blocks(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length,
    int64_t block_count
) {
    if (!sequences_packed.is_cuda() || !quartet_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (sequences_packed.dtype() != torch::kUInt8) {
        throw std::runtime_error("packed sequences must be torch.uint8");
    }
    if (quartet_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("quartet_indices must be torch.int64");
    }
    if (sequences_packed.dim() != 2) {
        throw std::runtime_error(
            "packed sequences must have shape (n_species, packed_seq_length)"
        );
    }
    if (quartet_indices.dim() != 2 || quartet_indices.size(1) != 4) {
        throw std::runtime_error("quartet_indices must have shape (n_quartets, 4)");
    }
    if (seq_length <= 0 || block_count <= 0 || block_count > seq_length) {
        throw std::runtime_error(
            "seq_length and block_count must satisfy 0 < block_count <= seq_length"
        );
    }
    return compute_pattern_frequencies_cuda_packed_blocks_backend_kernel(
        sequences_packed.contiguous(),
        quartet_indices.contiguous(),
        seq_length,
        block_count
    );
}

torch::Tensor sparse_profile_distances_cuda(
    torch::Tensor profiles,
    torch::Tensor pair_indices
) {
    if (!profiles.is_cuda() || !pair_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (profiles.dtype() != torch::kFloat32) {
        throw std::runtime_error("profiles must be torch.float32");
    }
    if (pair_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("pair_indices must be torch.int64");
    }
    if (profiles.dim() != 3 || profiles.size(2) != 4 || profiles.size(1) < 1) {
        throw std::runtime_error("profiles must have shape (nodes, sites, 4)");
    }
    if (pair_indices.dim() != 2 || pair_indices.size(1) != 2) {
        throw std::runtime_error("pair_indices must have shape (pairs, 2)");
    }
    return sparse_profile_distances_cuda_backend_kernel(
        profiles.contiguous(), pair_indices.contiguous()
    );
}

torch::Tensor sparse_marginalized_profile_distances_cuda(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor pair_indices
) {
    if (!profiles.is_cuda() || !site_mismatch_prior.is_cuda() ||
        !pair_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (profiles.dtype() != torch::kFloat32 ||
        site_mismatch_prior.dtype() != torch::kFloat32) {
        throw std::runtime_error("profiles/prior must be torch.float32");
    }
    if (pair_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("pair_indices must be torch.int64");
    }
    if (profiles.dim() != 3 || profiles.size(2) != 4 || profiles.size(1) < 1 ||
        site_mismatch_prior.dim() != 1 ||
        site_mismatch_prior.size(0) != profiles.size(1)) {
        throw std::runtime_error(
            "profiles/prior must have shape (nodes, sites, 4)/(sites)"
        );
    }
    if (pair_indices.dim() != 2 || pair_indices.size(1) != 2) {
        throw std::runtime_error("pair_indices must have shape (pairs, 2)");
    }
    return sparse_marginalized_profile_distances_cuda_backend_kernel(
        profiles.contiguous(), site_mismatch_prior.contiguous(),
        pair_indices.contiguous()
    );
}

torch::Tensor marginalized_profile_row_sums_cuda(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor aggregate_profile
) {
    if (!profiles.is_cuda() || !site_mismatch_prior.is_cuda() ||
        !aggregate_profile.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (profiles.dtype() != torch::kFloat32 ||
        site_mismatch_prior.dtype() != torch::kFloat64 ||
        aggregate_profile.dtype() != torch::kFloat64) {
        throw std::runtime_error(
            "profiles/prior/aggregate must be float32/float64/float64"
        );
    }
    if (profiles.dim() != 3 || profiles.size(2) != 4 || profiles.size(1) < 1 ||
        site_mismatch_prior.dim() != 1 ||
        site_mismatch_prior.size(0) != profiles.size(1) ||
        aggregate_profile.dim() != 2 ||
        aggregate_profile.size(0) != profiles.size(1) ||
        aggregate_profile.size(1) != 4) {
        throw std::runtime_error(
            "profiles/prior/aggregate have incompatible shapes"
        );
    }
    return marginalized_profile_row_sums_cuda_backend_kernel(
        profiles.contiguous(), site_mismatch_prior.contiguous(),
        aggregate_profile.contiguous()
    );
}

torch::Tensor sparse_coverage_profile_distances_cuda(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor site_frequency,
    torch::Tensor site_imputation_weight,
    torch::Tensor pair_indices
) {
    if (!profiles.is_cuda() || !site_mismatch_prior.is_cuda() ||
        !site_frequency.is_cuda() || !site_imputation_weight.is_cuda() ||
        !pair_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (profiles.dtype() != torch::kFloat32 ||
        site_mismatch_prior.dtype() != torch::kFloat32 ||
        site_frequency.dtype() != torch::kFloat32 ||
        site_imputation_weight.dtype() != torch::kFloat32 ||
        pair_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("invalid coverage-distance tensor dtypes");
    }
    const auto sites = profiles.size(1);
    if (profiles.dim() != 3 || profiles.size(2) != 4 || sites < 1 ||
        site_mismatch_prior.dim() != 1 || site_mismatch_prior.size(0) != sites ||
        site_frequency.dim() != 2 || site_frequency.size(0) != sites ||
        site_frequency.size(1) != 4 ||
        site_imputation_weight.dim() != 1 ||
        site_imputation_weight.size(0) != sites ||
        pair_indices.dim() != 2 || pair_indices.size(1) != 2) {
        throw std::runtime_error("invalid coverage-distance tensor shapes");
    }
    return sparse_coverage_profile_distances_cuda_backend_kernel(
        profiles.contiguous(), site_mismatch_prior.contiguous(),
        site_frequency.contiguous(), site_imputation_weight.contiguous(),
        pair_indices.contiguous()
    );
}

torch::Tensor coverage_profile_row_sums_cuda(
    torch::Tensor profiles,
    torch::Tensor site_mismatch_prior,
    torch::Tensor site_frequency,
    torch::Tensor site_imputation_weight,
    torch::Tensor aggregate_profile
) {
    if (!profiles.is_cuda() || !site_mismatch_prior.is_cuda() ||
        !site_frequency.is_cuda() || !site_imputation_weight.is_cuda() ||
        !aggregate_profile.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (profiles.dtype() != torch::kFloat32 ||
        site_mismatch_prior.dtype() != torch::kFloat64 ||
        site_frequency.dtype() != torch::kFloat64 ||
        site_imputation_weight.dtype() != torch::kFloat64 ||
        aggregate_profile.dtype() != torch::kFloat64) {
        throw std::runtime_error("invalid coverage-row tensor dtypes");
    }
    const auto sites = profiles.size(1);
    if (profiles.dim() != 3 || profiles.size(2) != 4 || sites < 1 ||
        site_mismatch_prior.dim() != 1 || site_mismatch_prior.size(0) != sites ||
        site_frequency.dim() != 2 || site_frequency.size(0) != sites ||
        site_frequency.size(1) != 4 ||
        site_imputation_weight.dim() != 1 ||
        site_imputation_weight.size(0) != sites ||
        aggregate_profile.dim() != 2 || aggregate_profile.size(0) != sites ||
        aggregate_profile.size(1) != 4) {
        throw std::runtime_error("invalid coverage-row tensor shapes");
    }
    return coverage_profile_row_sums_cuda_backend_kernel(
        profiles.contiguous(), site_mismatch_prior.contiguous(),
        site_frequency.contiguous(), site_imputation_weight.contiguous(),
        aggregate_profile.contiguous()
    );
}

torch::Tensor sparse_state_distances_cuda(
    torch::Tensor states,
    torch::Tensor pair_indices
) {
    if (!states.is_cuda() || !pair_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (states.dtype() != torch::kUInt8) {
        throw std::runtime_error("states must be torch.uint8");
    }
    if (pair_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("pair_indices must be torch.int64");
    }
    if (states.dim() != 2 || states.size(1) < 1) {
        throw std::runtime_error("states must have shape (nodes, sites)");
    }
    if (pair_indices.dim() != 2 || pair_indices.size(1) != 2) {
        throw std::runtime_error("pair_indices must have shape (pairs, 2)");
    }
    return sparse_state_distances_cuda_backend_kernel(
        states.contiguous(), pair_indices.contiguous()
    );
}

torch::Tensor sparse_imputed_state_distances_cuda(
    torch::Tensor states,
    torch::Tensor site_frequency,
    torch::Tensor pair_indices
) {
    if (!states.is_cuda() || !site_frequency.is_cuda() || !pair_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (states.dtype() != torch::kUInt8 || site_frequency.dtype() != torch::kFloat32) {
        throw std::runtime_error("states/site_frequency must be uint8/float32");
    }
    if (pair_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("pair_indices must be torch.int64");
    }
    if (states.dim() != 2 || site_frequency.dim() != 2 ||
        site_frequency.size(0) != states.size(1) || site_frequency.size(1) != 4) {
        throw std::runtime_error("site_frequency must have shape (sites, 4)");
    }
    if (pair_indices.dim() != 2 || pair_indices.size(1) != 2) {
        throw std::runtime_error("pair_indices must have shape (pairs, 2)");
    }
    return sparse_imputed_state_distances_cuda_backend_kernel(
        states.contiguous(), site_frequency.contiguous(), pair_indices.contiguous()
    );
}

void accumulate_sparse_profile_matches_cuda(
    torch::Tensor profiles,
    torch::Tensor pair_indices,
    torch::Tensor matches
) {
    if (!profiles.is_cuda() || !pair_indices.is_cuda() || !matches.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (profiles.dtype() != torch::kFloat32 ||
        pair_indices.dtype() != torch::kInt64 ||
        matches.dtype() != torch::kFloat64) {
        throw std::runtime_error("profiles/pairs/matches must be float32/int64/float64");
    }
    if (profiles.dim() != 3 || profiles.size(2) != 4 || profiles.size(1) < 1 ||
        pair_indices.dim() != 2 || pair_indices.size(1) != 2 ||
        matches.dim() != 1 || matches.size(0) != pair_indices.size(0)) {
        throw std::runtime_error("invalid streamed sparse-distance tensor shapes");
    }
    if (profiles.device() != pair_indices.device() ||
        profiles.device() != matches.device()) {
        throw std::runtime_error("streamed sparse-distance tensors must share a device");
    }
    accumulate_sparse_profile_matches_cuda_backend_kernel(
        profiles.contiguous(), pair_indices.contiguous(), matches
    );
}

torch::Tensor complete_state_projections_cuda(
    torch::Tensor states,
    torch::Tensor directions
) {
    if (!states.is_cuda() || !directions.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (states.dtype() != torch::kUInt8 || directions.dtype() != torch::kInt8) {
        throw std::runtime_error("states/directions must be uint8/int8");
    }
    if (states.dim() != 2 || directions.dim() != 3 ||
        directions.size(0) != states.size(1) || directions.size(1) != 4 ||
        directions.size(2) < 1) {
        throw std::runtime_error(
            "directions must have shape (sites, 4, projections)"
        );
    }
    return complete_state_projections_cuda_backend_kernel(
        states.contiguous(), directions.contiguous()
    );
}

torch::Tensor complete_state_row_sums_cuda(
    torch::Tensor states,
    torch::Tensor site_counts,
    bool float64_output
) {
    if (!states.is_cuda() || !site_counts.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (states.dtype() != torch::kUInt8 || site_counts.dtype() != torch::kInt64) {
        throw std::runtime_error("states/site_counts must be uint8/int64");
    }
    if (states.dim() != 2 || site_counts.dim() != 2 ||
        site_counts.size(0) != states.size(1) || site_counts.size(1) != 4) {
        throw std::runtime_error("site_counts must have shape (sites, 4)");
    }
    return complete_state_row_sums_cuda_backend_kernel(
        states.contiguous(), site_counts.contiguous(), float64_output
    );
}

torch::Tensor imputed_state_row_sums_cuda(
    torch::Tensor states,
    torch::Tensor site_frequency,
    torch::Tensor aggregate_profile,
    bool float64_output
) {
    if (!states.is_cuda() || !site_frequency.is_cuda() || !aggregate_profile.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    const auto expected_aggregate_dtype =
        float64_output ? torch::kFloat64 : torch::kFloat32;
    if (states.dtype() != torch::kUInt8 || site_frequency.dtype() != torch::kFloat32 ||
        aggregate_profile.dtype() != expected_aggregate_dtype) {
        throw std::runtime_error(
            "states/site_frequency must be uint8/float32 and aggregate_profile "
            "must match the requested output precision"
        );
    }
    if (states.dim() != 2 || site_frequency.dim() != 2 ||
        aggregate_profile.dim() != 2 ||
        site_frequency.size(0) != states.size(1) || site_frequency.size(1) != 4 ||
        aggregate_profile.sizes() != site_frequency.sizes()) {
        throw std::runtime_error(
            "site_frequency and aggregate_profile must have shape (sites, 4)"
        );
    }
    return imputed_state_row_sums_cuda_backend_kernel(
        states.contiguous(), site_frequency.contiguous(),
        aggregate_profile.contiguous(), float64_output
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compute_pattern_frequencies_cuda_packed",
        &compute_pattern_frequencies_cuda_packed,
        "Pattern frequency computation for packed sequences (CUDA)"
    );
    m.def(
        "compute_pattern_frequencies_cuda_packed_blocks",
        &compute_pattern_frequencies_cuda_packed_blocks,
        "Pattern frequencies over a fixed partition of one packed MSA (CUDA)"
    );
    m.def(
        "sparse_profile_distances_cuda",
        &sparse_profile_distances_cuda,
        "Sparse profile mismatch distances (CUDA)"
    );
    m.def(
        "sparse_marginalized_profile_distances_cuda",
        &sparse_marginalized_profile_distances_cuda,
        "Sparse profile distances with marginalized missing states (CUDA)"
    );
    m.def(
        "marginalized_profile_row_sums_cuda",
        &marginalized_profile_row_sums_cuda,
        "Exact aggregate row sums for marginalized profile distance (CUDA)"
    );
    m.def(
        "sparse_coverage_profile_distances_cuda",
        &sparse_coverage_profile_distances_cuda,
        "Sparse coverage-calibrated profile distances (CUDA)"
    );
    m.def(
        "coverage_profile_row_sums_cuda",
        &coverage_profile_row_sums_cuda,
        "Exact aggregate row sums for coverage-calibrated distance (CUDA)"
    );
    m.def(
        "sparse_state_distances_cuda",
        &sparse_state_distances_cuda,
        "Sparse complete-state mismatch distances (CUDA)"
    );
    m.def(
        "sparse_imputed_state_distances_cuda",
        &sparse_imputed_state_distances_cuda,
        "Sparse imputed-profile distances from compact states (CUDA)"
    );
    m.def(
        "accumulate_sparse_profile_matches_cuda",
        &accumulate_sparse_profile_matches_cuda,
        "Accumulate sparse profile matches from one site chunk (CUDA)"
    );
    m.def(
        "complete_state_projections_cuda",
        &complete_state_projections_cuda,
        "Exact signed projections from complete uint8 states (CUDA)"
    );
    m.def(
        "complete_state_row_sums_cuda",
        &complete_state_row_sums_cuda,
        "Exact complete-state mismatch row sums (CUDA)"
    );
    m.def(
        "imputed_state_row_sums_cuda",
        &imputed_state_row_sums_cuda,
        "Imputed-profile mismatch row sums from compact states (CUDA)"
    );
}
