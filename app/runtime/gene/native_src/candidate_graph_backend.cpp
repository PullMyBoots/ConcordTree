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


// Coordinate-preserving gather: the original row/name and column orders survive.
py::array_t<uint8_t> gather_states(
    py::array_t<uint8_t> states,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> columns,
    int workers) {
    if (states.ndim()!=2 || rows.ndim()!=1 || columns.ndim()!=1 || workers<1)
        throw std::runtime_error("invalid state gather input");
    const int64_t n=rows.size(), l=columns.size();
    const int64_t rs=states.strides(0), cs=states.strides(1);
    const auto* r=rows.data();const auto* c=columns.data();const auto* x=states.data();
    for (int64_t i=0;i<n;++i) if (r[i]<0 || r[i]>=states.shape(0)) throw std::runtime_error("row out of range");
    for (int64_t j=0;j<l;++j) if (c[j]<0 || c[j]>=states.shape(1)) throw std::runtime_error("site out of range");
    py::array_t<uint8_t> output({n,l});auto* y=output.mutable_data();
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(static) num_threads(workers)
      for(int64_t i=0;i<n;++i) for(int64_t j=0;j<l;++j) y[i*l+j]=x[r[i]*rs+c[j]*cs];
    }
    return output;
}

py::array_t<float> encode_profiles(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> states,
    py::array_t<float, py::array::c_style | py::array::forcecast> frequency,
    int workers) {
    if(states.ndim()!=2 || frequency.ndim()!=2 || frequency.shape(0)!=states.shape(1) || frequency.shape(1)!=4 || workers<1)
        throw std::runtime_error("invalid profile encoding input");
    const int64_t n=states.shape(0),l=states.shape(1);const auto* x=states.data();const auto* f=frequency.data();
    py::array_t<float> output({n,l,int64_t(4)});auto* y=output.mutable_data();
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(static) num_threads(workers)
      for(int64_t i=0;i<n;++i) for(int64_t j=0;j<l;++j) {
        const uint8_t b=x[i*l+j];
        for(int c=0;c<4;++c) y[(i*l+j)*4+c]=(b<4 ? float(b==c) : f[j*4+c]);
      }
    }
    return output;
}

py::array_t<float> pack_profile_sites(
    py::array_t<float, py::array::c_style | py::array::forcecast> raw,
    int64_t begin,int64_t end,int workers) {
    if(raw.ndim()!=3 || begin<0 || end<begin || end>raw.shape(1) || workers<1)
        throw std::runtime_error("invalid site packing input");
    const int64_t n=raw.shape(0),l=raw.shape(1),q=raw.shape(2),b=end-begin;
    py::array_t<float> output({n,b,q});auto* y=output.mutable_data();const auto* x=raw.data();
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(static) num_threads(workers)
      for(int64_t i=0;i<n;++i) std::memcpy(y+i*b*q,x+(i*l+begin)*q,size_t(b*q)*sizeof(float));
    }
    return output;
}

py::list merge_profile_pairs(py::sequence left,py::sequence right,int workers) {
    const int64_t n=py::len(left);
    if(py::len(right)!=n || workers<1) throw std::runtime_error("invalid merge lists");
    using Array=py::array_t<float,py::array::c_style>;
    std::vector<Array> inputs;inputs.reserve(n*2);
    std::vector<const float*> a(n),b(n);std::vector<float*> c(n);
    py::list outputs;int64_t width=0;
    for(int64_t i=0;i<n;++i) {
      Array x=Array::ensure(left[i]),y=Array::ensure(right[i]);
      if(!x || !y || x.ndim()!=2 || y.ndim()!=2 || x.shape(0)!=y.shape(0) || x.shape(1)!=y.shape(1))
        throw std::runtime_error("invalid merge profile");
      if(i && x.size()!=width) throw std::runtime_error("unequal merge sizes");
      width=x.size();inputs.push_back(x);inputs.push_back(y);a[i]=x.data();b[i]=y.data();
      py::array_t<float> z({x.shape(0),x.shape(1)});c[i]=z.mutable_data();outputs.append(z);
    }
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(static) num_threads(workers)
      for(int64_t i=0;i<n;++i) for(int64_t j=0;j<width;++j) c[i][j]=(a[i][j]+b[i][j])*0.5f;
    }
    return outputs;
}

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
        constexpr int64_t coordinate_tile = 256;
#ifdef _OPENMP
        omp_set_num_threads(workers);
#pragma omp parallel for schedule(static)
#endif
        // Tile independent coordinates, preserving ascending row addition
        // order for each double accumulator. Adjacent coordinates share each
        // cache line rather than making a strided pass for every coordinate.
        for (int64_t start = 0; start < width; start += coordinate_tile) {
            const int64_t count = std::min(coordinate_tile, width - start);
            std::array<double, coordinate_tile> totals{};
            for (int64_t row = 0; row < rows; ++row) {
                const float* values = raw + row * width + start;
                for (int64_t j = 0; j < count; ++j) {
                    totals[static_cast<size_t>(j)] += static_cast<double>(values[j]);
                }
            }
            for (int64_t j = 0; j < count; ++j) {
                aggregate[static_cast<size_t>(start + j)] = totals[static_cast<size_t>(j)];
            }
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

// The same margin/edge ordered greedy NNI as learned_nni.py. Keep one sorted
// neighbor table across passes, rather than reconstructing Python sets/objects.
py::array_t<int64_t> apply_nni_moves(
    py::array_t<int64_t,py::array::c_style> neighbors,
    py::array_t<int64_t,py::array::c_style> edges,
    py::array_t<int64_t,py::array::c_style> branches,
    py::array_t<int64_t,py::array::c_style> best,
    py::array_t<double,py::array::c_style> margins,double min_margin) {
    if(neighbors.ndim()!=2 || neighbors.shape(1)!=3 || !neighbors.writeable() ||
       edges.ndim()!=2 || edges.shape(1)!=2 || branches.ndim()!=2 || branches.shape(1)!=4 ||
       branches.shape(0)!=edges.shape(0) || best.ndim()!=1 || best.size()!=edges.shape(0) ||
       margins.ndim()!=1 || margins.size()!=edges.shape(0)) throw std::runtime_error("invalid NNI arrays");
    const int64_t n=neighbors.shape(0),m=edges.shape(0);
    auto* adj=neighbors.mutable_data();const auto* es=edges.data();const auto* bs=branches.data();
    const auto* which=best.data();const auto* gap=margins.data();
    std::vector<int64_t> order,selected;order.reserve(m);selected.reserve(m);
    for(int64_t i=0;i<m;++i) {
        if(which[i]<0 || which[i]>2) throw std::runtime_error("invalid NNI alternative");
        for(int j=0;j<2;++j) if(es[2*i+j]<0 || es[2*i+j]>=n) throw std::runtime_error("invalid NNI edge");
        for(int j=0;j<4;++j) if(bs[4*i+j]<0 || bs[4*i+j]>=n) throw std::runtime_error("invalid NNI branch");
        if(which[i]!=0 && gap[i]>=min_margin)order.push_back(i);
    }
    {
        py::gil_scoped_release release;
        std::sort(order.begin(),order.end(),[&](int64_t a,int64_t b){
            if(gap[a]!=gap[b])return gap[a]>gap[b];
            if(es[2*a]!=es[2*b])return es[2*a]<es[2*b];
            return es[2*a+1]<es[2*b+1];
        });
        std::vector<uint8_t> occupied(n,0);
        auto contains=[&](int64_t a,int64_t b){return std::find(adj+3*a,adj+3*a+3,b)!=adj+3*a+3;};
        auto replace=[&](int64_t a,int64_t old,int64_t fresh){
            auto* first=adj+3*a;auto* at=std::find(first,first+3,old);
            if(at==first+3)throw std::runtime_error("asymmetric NNI adjacency");
            *at=fresh;
            // Leaves keep the neighbor first and -1 padding last.
            std::sort(first,first+3,[](int64_t x,int64_t y){return x>=0 && (y<0 || x<y);});
        };
        for(auto i:order) {
            const int64_t u=es[2*i],v=es[2*i+1],b=bs[4*i+1],d=bs[4*i+(which[i]==1?2:3)];
            if(occupied[u] || occupied[v] || !contains(u,b) || !contains(v,d))continue;
            replace(u,b,d);replace(b,u,v);replace(v,d,b);replace(d,v,u);
            occupied[u]=occupied[v]=1;selected.push_back(i);
        }
    }
    py::array_t<int64_t> out(selected.size());std::copy(selected.begin(),selected.end(),out.mutable_data());return out;
}

PYBIND11_MODULE(candidate_graph_backend, module) {
    module.def("apply_nni_moves", &apply_nni_moves);
    module.def("gather_states", &gather_states);
    module.def("encode_profiles", &encode_profiles);
    module.def("pack_profile_sites", &pack_profile_sites);
    module.def("merge_profile_pairs", &merge_profile_pairs);
    module.def("stack_profile_rows", &stack_profile_rows);
    module.def("aggregate_mismatch_base_rows", &aggregate_mismatch_base_rows);
    module.def("projection_pairs", &projection_pairs);
    module.def("select_directed_pairs", &select_directed_pairs);
    module.def("select_nj_merges", &select_nj_merges);
}
