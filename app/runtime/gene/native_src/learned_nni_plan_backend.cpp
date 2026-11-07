#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Group = std::vector<int64_t>;

struct EdgePlan {
    int64_t left;
    int64_t right;
    std::array<int64_t, 4> branch_nodes;
    std::array<Group, 4> groups;
    int64_t quartet_count;
};

Group near_representatives(
    const py::detail::unchecked_reference<int64_t, 2>& neighbors,
    int64_t start,
    int64_t blocked,
    int64_t n_taxa,
    int64_t limit
) {
    std::deque<std::pair<int64_t, int64_t>> queue;
    queue.emplace_back(start, blocked);
    Group leaves;
    leaves.reserve(static_cast<std::size_t>(limit));
    while (!queue.empty() && static_cast<int64_t>(leaves.size()) < limit) {
        const auto [node, parent] = queue.front();
        queue.pop_front();
        if (node < n_taxa) {
            leaves.push_back(node);
            continue;
        }
        for (py::ssize_t column = 0; column < neighbors.shape(1); ++column) {
            const int64_t child = neighbors(node, column);
            if (child < 0) {
                break;
            }
            if (child != parent) {
                queue.emplace_back(child, node);
            }
        }
    }
    return leaves;
}

}  // namespace

py::tuple compile_nni_plan(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> neighbor_array,
    int64_t n_taxa,
    int64_t representatives
) {
    if (neighbor_array.ndim() != 2 || neighbor_array.shape(1) != 3) {
        throw std::runtime_error("neighbors must have shape (nodes, 3)");
    }
    if (n_taxa < 4 || n_taxa > neighbor_array.shape(0)) {
        throw std::runtime_error("invalid n_taxa");
    }
    if (representatives < 1) {
        throw std::runtime_error("representatives must be positive");
    }
    const auto neighbors = neighbor_array.unchecked<2>();
    for (py::ssize_t node = 0; node < neighbor_array.shape(0); ++node) {
        int degree = 0;
        int64_t previous = -1;
        for (py::ssize_t column = 0; column < 3; ++column) {
            const int64_t neighbor = neighbors(node, column);
            if (neighbor < 0) {
                continue;
            }
            if (neighbor >= neighbor_array.shape(0) || neighbor <= previous) {
                throw std::runtime_error("neighbor rows must be sorted and valid");
            }
            previous = neighbor;
            ++degree;
        }
        if ((node < n_taxa && degree != 1) || (node >= n_taxa && degree != 3)) {
            throw std::runtime_error("neighbors do not encode an unrooted binary tree");
        }
    }

    std::vector<EdgePlan> plans;
    int64_t total_quartets = 0;
    for (int64_t left = n_taxa; left < neighbor_array.shape(0); ++left) {
        for (py::ssize_t column = 0; column < 3; ++column) {
            const int64_t right = neighbors(left, column);
            if (right <= left || right < n_taxa) {
                continue;
            }
            EdgePlan plan;
            plan.left = left;
            plan.right = right;
            int left_count = 0;
            int right_count = 0;
            for (py::ssize_t index = 0; index < 3; ++index) {
                const int64_t node = neighbors(left, index);
                if (node != right) {
                    plan.branch_nodes[left_count++] = node;
                }
            }
            for (py::ssize_t index = 0; index < 3; ++index) {
                const int64_t node = neighbors(right, index);
                if (node != left) {
                    plan.branch_nodes[2 + right_count++] = node;
                }
            }
            if (left_count != 2 || right_count != 2) {
                throw std::runtime_error("internal edge does not have four branches");
            }
            for (int group = 0; group < 4; ++group) {
                const int64_t owner = group < 2 ? left : right;
                plan.groups[group] = near_representatives(
                    neighbors,
                    plan.branch_nodes[group],
                    owner,
                    n_taxa,
                    representatives
                );
                if (plan.groups[group].empty()) {
                    throw std::runtime_error("directed branch contains no taxon");
                }
            }
            plan.quartet_count = 1;
            for (const auto& group : plan.groups) {
                plan.quartet_count *= static_cast<int64_t>(group.size());
            }
            total_quartets += plan.quartet_count;
            plans.push_back(std::move(plan));
        }
    }

    py::array_t<int64_t> edges(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(plans.size()), static_cast<py::ssize_t>(2)
    });
    py::array_t<int64_t> branches(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(plans.size()), static_cast<py::ssize_t>(4)
    });
    py::array_t<int64_t> offsets(static_cast<py::ssize_t>(plans.size() + 1));
    py::array_t<int64_t> ordered(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(total_quartets), static_cast<py::ssize_t>(4)
    });
    py::array_t<int64_t> canonical(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(total_quartets), static_cast<py::ssize_t>(4)
    });
    auto edge_out = edges.mutable_unchecked<2>();
    auto branch_out = branches.mutable_unchecked<2>();
    auto offset_out = offsets.mutable_unchecked<1>();
    auto ordered_out = ordered.mutable_unchecked<2>();
    auto canonical_out = canonical.mutable_unchecked<2>();

    int64_t row = 0;
    offset_out(0) = 0;
    for (std::size_t edge_index = 0; edge_index < plans.size(); ++edge_index) {
        const auto& plan = plans[edge_index];
        edge_out(edge_index, 0) = plan.left;
        edge_out(edge_index, 1) = plan.right;
        for (int group = 0; group < 4; ++group) {
            branch_out(edge_index, group) = plan.branch_nodes[group];
        }
        for (const int64_t one : plan.groups[0]) {
            for (const int64_t two : plan.groups[1]) {
                for (const int64_t three : plan.groups[2]) {
                    for (const int64_t four : plan.groups[3]) {
                        std::array<int64_t, 4> values = {one, two, three, four};
                        for (int column = 0; column < 4; ++column) {
                            ordered_out(row, column) = values[column];
                        }
                        std::sort(values.begin(), values.end());
                        for (int column = 0; column < 4; ++column) {
                            canonical_out(row, column) = values[column];
                        }
                        ++row;
                    }
                }
            }
        }
        offset_out(edge_index + 1) = row;
    }
    if (row != total_quartets) {
        throw std::runtime_error("compiled quartet count mismatch");
    }
    return py::make_tuple(edges, branches, offsets, ordered, canonical);
}

py::array_t<double> reduce_nni_arithmetic(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ordered,
    py::array_t<double, py::array::c_style | py::array::forcecast> probabilities,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> offsets
) {
    if (ordered.ndim() != 2 || ordered.shape(1) != 4 ||
        probabilities.ndim() != 2 || probabilities.shape(1) != 3 ||
        probabilities.shape(0) != ordered.shape(0) || offsets.ndim() != 1 ||
        offsets.shape(0) < 2 || offsets.at(0) != 0 ||
        offsets.at(offsets.shape(0) - 1) != ordered.shape(0)) {
        throw std::runtime_error("invalid ordered quartet, probability, or offset arrays");
    }
    const int64_t edges = offsets.shape(0) - 1;
    py::array_t<double> output(std::vector<py::ssize_t>{edges, 3});
    const auto quartet = ordered.unchecked<2>();
    const auto posterior = probabilities.unchecked<2>();
    const auto starts = offsets.unchecked<1>();
    auto scores = output.mutable_unchecked<2>();
    constexpr int8_t pair_class[4][4] = {
        {-1, 0, 1, 2},
        {0, -1, 2, 1},
        {1, 2, -1, 0},
        {2, 1, 0, -1},
    };
    {
        py::gil_scoped_release release;
        for (int64_t edge = 0; edge < edges; ++edge) {
            const int64_t begin = starts(edge);
            const int64_t end = starts(edge + 1);
            if (end <= begin) {
                throw std::runtime_error("each NNI edge must contain quartet rows");
            }
            double sums[3] = {0.0, 0.0, 0.0};
            for (int64_t row = begin; row < end; ++row) {
                int ranks[4] = {0, 0, 0, 0};
                for (int left = 0; left < 4; ++left) {
                    for (int right = 0; right < 4; ++right) {
                        ranks[left] += quartet(row, left) > quartet(row, right);
                    }
                }
                for (int column = 0; column < 3; ++column) {
                    const int cls = pair_class[ranks[0]][ranks[column + 1]];
                    if (cls < 0) {
                        throw std::runtime_error("ordered quartet taxa must be distinct");
                    }
                    sums[column] += posterior(row, cls);
                }
            }
            const double inverse = 1.0 / static_cast<double>(end - begin);
            for (int column = 0; column < 3; ++column) {
                scores(edge, column) = sums[column] * inverse;
            }
        }
    }
    return output;
}

PYBIND11_MODULE(learned_nni_plan_backend, module) {
    module.def(
        "reduce_nni_arithmetic",
        &reduce_nni_arithmetic,
        "Fuse quartet-class remapping and arithmetic edge aggregation"
    );
    module.def(
        "compile_nni_plan",
        &compile_nni_plan,
        py::arg("neighbors"),
        py::arg("n_taxa"),
        py::arg("representatives") = 4,
        "Compile exact learned-NNI edge metadata and ordered quartet rows"
    );
}
