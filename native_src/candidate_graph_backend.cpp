#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

py::array_t<float> stack_profile_rows(py::sequence profiles, int workers) {
    const py::ssize_t rows = py::len(profiles);
    if (rows < 1 || workers <= 0) {
        throw std::runtime_error("profile sequence must be nonempty and workers positive");
    }
    using FloatArray = py::array_t<float, py::array::c_style>;
    std::vector<FloatArray> arrays;
    std::vector<const float*> inputs;
    arrays.reserve(static_cast<std::size_t>(rows));
    inputs.reserve(static_cast<std::size_t>(rows));
    py::ssize_t sites = -1;
    py::ssize_t states = -1;
    for (py::ssize_t row = 0; row < rows; ++row) {
        FloatArray array = FloatArray::ensure(profiles[row]);
        if (!array || array.ndim() != 2) {
            throw std::runtime_error("profile rows must be contiguous float32 matrices");
        }
        if (row == 0) {
            sites = array.shape(0);
            states = array.shape(1);
            if (sites < 1 || states < 1) {
                throw std::runtime_error("profile rows must have positive dimensions");
            }
        } else if (array.shape(0) != sites || array.shape(1) != states) {
            throw std::runtime_error("profile rows must have equal dimensions");
        }
        arrays.push_back(array);
        inputs.push_back(array.data());
    }
    py::array_t<float> output({rows, sites, states});
    float* target = output.mutable_data();
    const std::size_t width = static_cast<std::size_t>(sites * states);
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        omp_set_num_threads(workers);
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t row = 0; row < rows; ++row) {
            std::memcpy(
                target + static_cast<std::size_t>(row) * width,
                inputs[static_cast<std::size_t>(row)],
                width * sizeof(float)
            );
        }
    }
    return output;
}

py::array_t<double> aggregate_mismatch_base_rows(
    py::array_t<float, py::array::c_style | py::array::forcecast> profiles,
    int workers
) {
    if (profiles.ndim() != 3 || profiles.shape(0) < 1 ||
        profiles.shape(1) < 1 || profiles.shape(2) < 1 || workers <= 0) {
        throw std::runtime_error("invalid profile slab or worker count");
    }
    const int64_t rows = profiles.shape(0);
    const int64_t sites = profiles.shape(1);
    const int64_t states = profiles.shape(2);
    const int64_t width = sites * states;
    const float* raw = profiles.data();
    std::vector<double> aggregate(static_cast<size_t>(width), 0.0);
    py::array_t<double> output(rows);
    double* result = output.mutable_data();
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        omp_set_num_threads(workers);
#pragma omp parallel for schedule(static)
#endif
        for (int64_t coordinate = 0; coordinate < width; ++coordinate) {
            double total = 0.0;
            for (int64_t row = 0; row < rows; ++row) {
                total += static_cast<double>(raw[row * width + coordinate]);
            }
            aggregate[static_cast<size_t>(coordinate)] = total;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t row = 0; row < rows; ++row) {
            const float* values = raw + row * width;
            double dot_all = 0.0;
            double dot_self = 0.0;
            for (int64_t coordinate = 0; coordinate < width; ++coordinate) {
                const double value = static_cast<double>(values[coordinate]);
                dot_all += value * aggregate[static_cast<size_t>(coordinate)];
                dot_self += value * value;
            }
            const double base_all = static_cast<double>(rows) - dot_all / sites;
            const double base_self = 1.0 - dot_self / sites;
            result[row] = base_all - base_self;
        }
    }
    return output;
}

py::array_t<int64_t> projection_pairs(
    py::array_t<float, py::array::c_style | py::array::forcecast> projected,
    int window,
    int workers
) {
    if (projected.ndim() != 2 || projected.shape(0) < 2 || window <= 0 || workers <= 0) {
        throw std::runtime_error("invalid projected array, window, or worker count");
    }
    const int64_t n = projected.shape(0);
    const int64_t columns = projected.shape(1);
    const int64_t width = std::min<int64_t>(window, n - 1);
    const auto values = projected.unchecked<2>();
    std::vector<std::vector<int64_t>> column_keys(static_cast<size_t>(columns));
#ifdef _OPENMP
    omp_set_num_threads(workers);
#pragma omp parallel for schedule(static)
#endif
    for (int64_t column = 0; column < columns; ++column) {
        std::vector<int64_t> order(static_cast<size_t>(n));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::stable_sort(order.begin(), order.end(), [&](int64_t left, int64_t right) {
            return values(left, column) < values(right, column);
        });
        auto& keys = column_keys[static_cast<size_t>(column)];
        keys.reserve(static_cast<size_t>(width * n - width * (width + 1) / 2));
        for (int64_t offset = 1; offset <= width; ++offset) {
            for (int64_t rank = 0; rank + offset < n; ++rank) {
                const int64_t one = order[static_cast<size_t>(rank)];
                const int64_t two = order[static_cast<size_t>(rank + offset)];
                keys.push_back(std::min(one, two) * n + std::max(one, two));
            }
        }
    }
    size_t total = 0;
    for (const auto& keys : column_keys) {
        total += keys.size();
    }
    std::vector<int64_t> keys;
    keys.reserve(total);
    for (auto& one_column : column_keys) {
        keys.insert(keys.end(), one_column.begin(), one_column.end());
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    py::array_t<int64_t> output({static_cast<py::ssize_t>(keys.size()), py::ssize_t{2}});
    auto pairs = output.mutable_unchecked<2>();
    for (size_t row = 0; row < keys.size(); ++row) {
        pairs(static_cast<py::ssize_t>(row), 0) = keys[row] / n;
        pairs(static_cast<py::ssize_t>(row), 1) = keys[row] % n;
    }
    return output;
}

py::array_t<int64_t> select_directed_pairs(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> pair_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tie_rank,
    int cap,
    int workers
) {
    if (pair_array.ndim() != 2 || pair_array.shape(1) != 2 ||
        distances.ndim() != 1 || distances.shape(0) != pair_array.shape(0) ||
        tie_rank.ndim() != 1 || cap <= 0 || workers <= 0) {
        throw std::runtime_error("invalid pair, distance, tie-rank, cap, or worker input");
    }
    const int64_t n = tie_rank.shape(0);
    const auto pairs = pair_array.unchecked<2>();
    const auto edge_distance = distances.unchecked<1>();
    const auto ranks = tie_rank.unchecked<1>();
    std::vector<std::vector<int64_t>> incident(static_cast<size_t>(n));
    for (int64_t row = 0; row < pair_array.shape(0); ++row) {
        const int64_t left = pairs(row, 0);
        const int64_t right = pairs(row, 1);
        if (left < 0 || right <= left || right >= n) {
            throw std::runtime_error("pair rows must be canonical and in range");
        }
        incident[static_cast<size_t>(left)].push_back(row);
        incident[static_cast<size_t>(right)].push_back(row);
    }
#ifdef _OPENMP
    omp_set_num_threads(workers);
#pragma omp parallel for schedule(dynamic, 32)
#endif
    for (int64_t owner = 0; owner < n; ++owner) {
        auto& rows = incident[static_cast<size_t>(owner)];
        std::sort(rows.begin(), rows.end(), [&](int64_t first, int64_t second) {
            if (edge_distance(first) != edge_distance(second)) {
                return edge_distance(first) < edge_distance(second);
            }
            const int64_t first_other =
                pairs(first, 0) == owner ? pairs(first, 1) : pairs(first, 0);
            const int64_t second_other =
                pairs(second, 0) == owner ? pairs(second, 1) : pairs(second, 0);
            return ranks(first_other) < ranks(second_other);
        });
    }
    std::vector<int64_t> starts(static_cast<size_t>(n + 1), 0);
    for (int64_t owner = 0; owner < n; ++owner) {
        starts[static_cast<size_t>(owner + 1)] = starts[static_cast<size_t>(owner)] +
            std::min<int64_t>(cap, incident[static_cast<size_t>(owner)].size());
    }
    py::array_t<int64_t> output(
        {static_cast<py::ssize_t>(starts.back()), py::ssize_t{2}}
    );
    auto selected = output.mutable_unchecked<2>();
    for (int64_t owner = 0; owner < n; ++owner) {
        const auto& rows = incident[static_cast<size_t>(owner)];
        const int64_t count = std::min<int64_t>(cap, rows.size());
        for (int64_t local = 0; local < count; ++local) {
            const int64_t pair_row = rows[static_cast<size_t>(local)];
            const int64_t other = pairs(pair_row, 0) == owner
                ? pairs(pair_row, 1) : pairs(pair_row, 0);
            const int64_t output_row = starts[static_cast<size_t>(owner)] + local;
            selected(output_row, 0) = owner;
            selected(output_row, 1) = other;
        }
    }
    return output;
}

py::tuple select_nj_merges(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> pair_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tie_rank,
    py::array_t<double, py::array::c_style | py::array::forcecast> row_sums,
    py::array_t<double, py::array::c_style | py::array::forcecast> offsets,
    int cap,
    int maximum_merges,
    int workers
) {
    if (pair_array.ndim() != 2 || pair_array.shape(1) != 2 ||
        distances.ndim() != 1 || distances.shape(0) != pair_array.shape(0) ||
        tie_rank.ndim() != 1 || row_sums.ndim() != 1 || offsets.ndim() != 1 ||
        row_sums.shape(0) != tie_rank.shape(0) ||
        offsets.shape(0) != tie_rank.shape(0) ||
        cap <= 0 || maximum_merges <= 0 || workers <= 0) {
        throw std::runtime_error(
            "invalid pair, distance, rank, row-sum, offset, cap, merge, or worker input"
        );
    }
    const int64_t n = tie_rank.shape(0);
    if (n < 4) {
        throw std::runtime_error("ordered NJ fold requires at least four active nodes");
    }
    const auto pairs = pair_array.unchecked<2>();
    const auto edge_distance = distances.unchecked<1>();
    const auto ranks = tie_rank.unchecked<1>();
    const auto rows = row_sums.unchecked<1>();
    const auto node_offsets = offsets.unchecked<1>();

    struct RankedEdge {
        int64_t pool_row;
        int64_t left;
        int64_t right;
        double q;
    };
    std::vector<std::array<int64_t, 3>> accepted;
    int64_t candidate_count = 0;
    int64_t maximum_degree = 0;
    {
        py::gil_scoped_release release;
        std::vector<std::vector<int64_t>> incident(static_cast<size_t>(n));
        for (int64_t row = 0; row < pair_array.shape(0); ++row) {
            const int64_t left = pairs(row, 0);
            const int64_t right = pairs(row, 1);
            if (left < 0 || right <= left || right >= n) {
                throw std::runtime_error("pair rows must be canonical and in range");
            }
            incident[static_cast<size_t>(left)].push_back(row);
            incident[static_cast<size_t>(right)].push_back(row);
        }
#ifdef _OPENMP
        omp_set_num_threads(workers);
#pragma omp parallel for schedule(dynamic, 32)
#endif
        for (int64_t owner = 0; owner < n; ++owner) {
            auto& owned_rows = incident[static_cast<size_t>(owner)];
            std::sort(owned_rows.begin(), owned_rows.end(), [&](int64_t first, int64_t second) {
                if (edge_distance(first) != edge_distance(second)) {
                    return edge_distance(first) < edge_distance(second);
                }
                const int64_t first_other =
                    pairs(first, 0) == owner ? pairs(first, 1) : pairs(first, 0);
                const int64_t second_other =
                    pairs(second, 0) == owner ? pairs(second, 1) : pairs(second, 0);
                if (ranks(first_other) != ranks(second_other)) {
                    return ranks(first_other) < ranks(second_other);
                }
                return first_other < second_other;
            });
        }

        std::vector<uint8_t> retained(static_cast<size_t>(pair_array.shape(0)), 0);
        for (int64_t owner = 0; owner < n; ++owner) {
            const auto& owned_rows = incident[static_cast<size_t>(owner)];
            const int64_t count = std::min<int64_t>(cap, owned_rows.size());
            maximum_degree = std::max(maximum_degree, count);
            for (int64_t local = 0; local < count; ++local) {
                retained[static_cast<size_t>(owned_rows[static_cast<size_t>(local)])] = 1;
            }
        }

        std::vector<RankedEdge> candidates;
        candidates.reserve(static_cast<size_t>(pair_array.shape(0)));
        for (int64_t pool_row = 0; pool_row < pair_array.shape(0); ++pool_row) {
            if (!retained[static_cast<size_t>(pool_row)]) {
                continue;
            }
            int64_t left = pairs(pool_row, 0);
            int64_t right = pairs(pool_row, 1);
            if (ranks(right) < ranks(left)) {
                std::swap(left, right);
            }
            double adjusted = edge_distance(pool_row);
            adjusted = adjusted + node_offsets(left);
            adjusted = adjusted + node_offsets(right);
            double q = static_cast<double>(n - 2) * adjusted;
            q = q - rows(left);
            q = q - rows(right);
            candidates.push_back({pool_row, left, right, q});
        }
        candidate_count = static_cast<int64_t>(candidates.size());
        if (candidates.empty()) {
            throw std::runtime_error("sparse candidate graph has no edges");
        }

        std::vector<int64_t> best(static_cast<size_t>(n), -1);
        auto better_for_owner = [&](int64_t candidate, int64_t incumbent, int64_t owner) {
            if (incumbent < 0) {
                return true;
            }
            const auto& one = candidates[static_cast<size_t>(candidate)];
            const auto& two = candidates[static_cast<size_t>(incumbent)];
            if (one.q != two.q) {
                return one.q < two.q;
            }
            const int64_t one_other = one.left == owner ? one.right : one.left;
            const int64_t two_other = two.left == owner ? two.right : two.left;
            if (ranks(one_other) != ranks(two_other)) {
                return ranks(one_other) < ranks(two_other);
            }
            return one_other < two_other;
        };
        for (int64_t index = 0; index < candidate_count; ++index) {
            const auto& edge = candidates[static_cast<size_t>(index)];
            if (better_for_owner(index, best[static_cast<size_t>(edge.left)], edge.left)) {
                best[static_cast<size_t>(edge.left)] = index;
            }
            if (better_for_owner(index, best[static_cast<size_t>(edge.right)], edge.right)) {
                best[static_cast<size_t>(edge.right)] = index;
            }
        }

        std::vector<int64_t> order(static_cast<size_t>(candidate_count));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::sort(order.begin(), order.end(), [&](int64_t first, int64_t second) {
            const auto& one = candidates[static_cast<size_t>(first)];
            const auto& two = candidates[static_cast<size_t>(second)];
            if (one.q != two.q) {
                return one.q < two.q;
            }
            if (ranks(one.left) != ranks(two.left)) {
                return ranks(one.left) < ranks(two.left);
            }
            if (ranks(one.right) != ranks(two.right)) {
                return ranks(one.right) < ranks(two.right);
            }
            if (one.left != two.left) {
                return one.left < two.left;
            }
            return one.right < two.right;
        });

        std::vector<uint8_t> consumed(static_cast<size_t>(n), 0);
        accepted.reserve(static_cast<size_t>(std::min<int64_t>(maximum_merges, n / 2)));
        for (int64_t candidate : order) {
            if (static_cast<int>(accepted.size()) >= maximum_merges) {
                break;
            }
            const auto& edge = candidates[static_cast<size_t>(candidate)];
            if (consumed[static_cast<size_t>(edge.left)] ||
                consumed[static_cast<size_t>(edge.right)]) {
                continue;
            }
            if (best[static_cast<size_t>(edge.left)] == candidate &&
                best[static_cast<size_t>(edge.right)] == candidate) {
                accepted.push_back({edge.left, edge.right, edge.pool_row});
                consumed[static_cast<size_t>(edge.left)] = 1;
                consumed[static_cast<size_t>(edge.right)] = 1;
            }
        }
        if (accepted.empty()) {
            const auto& edge = candidates[static_cast<size_t>(order.front())];
            accepted.push_back({edge.left, edge.right, edge.pool_row});
        }
    }

    py::array_t<int64_t> output(
        {static_cast<py::ssize_t>(accepted.size()), py::ssize_t{3}}
    );
    auto selected = output.mutable_unchecked<2>();
    for (size_t row = 0; row < accepted.size(); ++row) {
        selected(static_cast<py::ssize_t>(row), 0) = accepted[row][0];
        selected(static_cast<py::ssize_t>(row), 1) = accepted[row][1];
        selected(static_cast<py::ssize_t>(row), 2) = accepted[row][2];
    }
    return py::make_tuple(output, candidate_count, maximum_degree);
}

PYBIND11_MODULE(candidate_graph_backend, module) {
    module.def("stack_profile_rows", &stack_profile_rows);
    module.def("aggregate_mismatch_base_rows", &aggregate_mismatch_base_rows);
    module.def("projection_pairs", &projection_pairs);
    module.def("select_directed_pairs", &select_directed_pairs);
    module.def("select_nj_merges", &select_nj_merges);
}
