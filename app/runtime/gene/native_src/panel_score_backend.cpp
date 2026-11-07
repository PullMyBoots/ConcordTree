#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

int8_t pair_class(int left_rank, int right_rank) {
    static constexpr int8_t classes[4][4] = {
        {-1, 0, 1, 2},
        {0, -1, 2, 1},
        {1, 2, -1, 0},
        {2, 1, 0, -1},
    };
    return classes[left_rank][right_rank];
}

bool in_component(int64_t tick, int64_t entered, int64_t exited, bool complement) {
    const bool inside = entered <= tick && tick <= exited;
    return inside != complement;
}

double median_copy(const double* data, py::ssize_t size) {
    if (size <= 0) {
        throw std::runtime_error("median input must be non-empty");
    }
    std::vector<double> work(data, data + size);
    const auto middle = work.begin() + size / 2;
    std::nth_element(work.begin(), middle, work.end());
    const double upper = *middle;
    if (size % 2 != 0) {
        return upper;
    }
    const double lower = *std::max_element(work.begin(), middle);
    return (lower + upper) / 2.0;
}

using LeafDistance = std::pair<int32_t, int32_t>;

std::vector<LeafDistance> closest_leaves(
    const std::vector<LeafDistance>& candidates,
    int32_t limit
) {
    std::vector<LeafDistance> unique;
    unique.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        auto found = std::find_if(
            unique.begin(), unique.end(),
            [&](const LeafDistance& value) { return value.second == candidate.second; }
        );
        if (found == unique.end()) {
            unique.push_back(candidate);
        } else if (candidate.first < found->first) {
            found->first = candidate.first;
        }
    }
    std::sort(unique.begin(), unique.end());
    if (static_cast<int32_t>(unique.size()) > limit) {
        unique.resize(limit);
    }
    return unique;
}

int32_t tree_distance(
    int32_t left,
    int32_t right,
    const py::detail::unchecked_reference<int32_t, 1>& depth,
    const py::detail::unchecked_reference<int32_t, 2>& ancestors
) {
    const int32_t original_left_depth = depth(left);
    const int32_t original_right_depth = depth(right);
    if (depth(left) < depth(right)) {
        const int32_t temporary = left;
        left = right;
        right = temporary;
    }
    const int32_t difference = depth(left) - depth(right);
    for (py::ssize_t level = 0; level < ancestors.shape(0); ++level) {
        if ((difference >> level) & 1) {
            left = ancestors(level, left);
        }
    }
    if (left != right) {
        for (py::ssize_t level = ancestors.shape(0); level-- > 0;) {
            const int32_t lifted_left = ancestors(level, left);
            const int32_t lifted_right = ancestors(level, right);
            if (lifted_left != lifted_right) {
                left = lifted_left;
                right = lifted_right;
            }
        }
        left = ancestors(0, left);
    }
    return original_left_depth + original_right_depth - 2 * depth(left);
}

}  // namespace

py::array_t<int32_t> compile_directed_nearest_messages(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> edges,
    int32_t node_count,
    int32_t taxon_count,
    int32_t limit
) {
    if (edges.ndim() != 2 || edges.shape(1) != 2) {
        throw std::runtime_error("edges must have shape (undirected_edges, 2)");
    }
    if (node_count <= 0 || taxon_count <= 0 || taxon_count > node_count) {
        throw std::runtime_error("invalid node or taxon count");
    }
    if (limit <= 0) {
        throw std::runtime_error("limit must be positive");
    }
    const auto pairs = edges.unchecked<2>();
    std::vector<std::vector<int32_t>> adjacency(node_count);
    for (py::ssize_t row = 0; row < edges.shape(0); ++row) {
        const int32_t left = pairs(row, 0);
        const int32_t right = pairs(row, 1);
        if (left < 0 || left >= node_count || right < 0 || right >= node_count ||
            left == right) {
            throw std::runtime_error("edge endpoint is outside the tree");
        }
        adjacency[left].push_back(right);
        adjacency[right].push_back(left);
    }

    std::vector<std::vector<int32_t>> rows;
    {
        py::gil_scoped_release release;
        int32_t root = -1;
        int64_t degree_sum = 0;
        for (int32_t node = 0; node < node_count; ++node) {
            auto& neighbors = adjacency[node];
            std::sort(neighbors.begin(), neighbors.end());
            if (std::adjacent_find(neighbors.begin(), neighbors.end()) != neighbors.end()) {
                throw std::runtime_error("tree contains a duplicate edge");
            }
            if (neighbors.size() > 3) {
                throw std::runtime_error("nearest messages require a binary tree");
            }
            if (!neighbors.empty() && root == -1) {
                root = node;
            }
            degree_sum += static_cast<int64_t>(neighbors.size());
        }
        if (root == -1) {
            throw std::runtime_error("nearest messages require at least one edge");
        }

        std::vector<int32_t> parent(node_count, -2);
        std::vector<int32_t> order;
        order.reserve(node_count);
        parent[root] = -1;
        order.push_back(root);
        for (size_t offset = 0; offset < order.size(); ++offset) {
            const int32_t node = order[offset];
            for (const int32_t neighbor : adjacency[node]) {
                if (neighbor == parent[node]) {
                    continue;
                }
                if (parent[neighbor] != -2) {
                    throw std::runtime_error("adjacency is not a tree");
                }
                parent[neighbor] = node;
                order.push_back(neighbor);
            }
        }
        if (static_cast<int32_t>(order.size()) != node_count ||
            edges.shape(0) != node_count - 1) {
            throw std::runtime_error("adjacency is disconnected");
        }

        std::vector<std::vector<LeafDistance>> down(node_count);
        for (auto position = order.rbegin(); position != order.rend(); ++position) {
            const int32_t node = *position;
            std::vector<LeafDistance> candidates;
            candidates.reserve(1 + 2 * limit);
            if (node < taxon_count) {
                candidates.emplace_back(0, node);
            }
            for (const int32_t child : adjacency[node]) {
                if (parent[child] != node) {
                    continue;
                }
                for (const auto& value : down[child]) {
                    candidates.emplace_back(value.first + 1, value.second);
                }
            }
            down[node] = closest_leaves(candidates, limit);
            if (down[node].empty()) {
                throw std::runtime_error("rooted subtree contains no taxon");
            }
        }

        std::vector<std::vector<LeafDistance>> outside(node_count);
        rows.reserve(static_cast<size_t>(degree_sum));
        const auto emit = [&](int32_t owner, int32_t neighbor,
                              const std::vector<LeafDistance>& values) {
            std::vector<int32_t> row(2 + limit, -1);
            row[0] = owner;
            row[1] = neighbor;
            for (size_t index = 0; index < values.size(); ++index) {
                row[2 + index] = values[index].second;
            }
            rows.push_back(std::move(row));
        };
        for (const int32_t node : order) {
            for (const int32_t child : adjacency[node]) {
                if (parent[child] != node) {
                    continue;
                }
                std::vector<LeafDistance> candidates = outside[node];
                if (node < taxon_count) {
                    candidates.emplace_back(0, node);
                }
                for (const int32_t sibling : adjacency[node]) {
                    if (sibling == child || parent[sibling] != node) {
                        continue;
                    }
                    for (const auto& value : down[sibling]) {
                        candidates.emplace_back(value.first + 1, value.second);
                    }
                }
                const auto side = closest_leaves(candidates, limit);
                if (side.empty()) {
                    throw std::runtime_error("directed side contains no taxon");
                }
                emit(node, child, side);
                emit(child, node, down[child]);
                outside[child].reserve(side.size());
                for (const auto& value : side) {
                    outside[child].emplace_back(value.first + 1, value.second);
                }
            }
        }
        if (static_cast<int64_t>(rows.size()) != degree_sum) {
            throw std::runtime_error("native directed-message count mismatch");
        }
    }

    py::array_t<int32_t> output({static_cast<py::ssize_t>(rows.size()),
                                 static_cast<py::ssize_t>(2 + limit)});
    auto result = output.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < output.shape(0); ++row) {
        for (py::ssize_t column = 0; column < output.shape(1); ++column) {
            result(row, column) = rows[row][column];
        }
    }
    return output;
}

py::list compile_contextual_panel_covers(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> edges,
    int32_t node_count,
    int32_t taxon_count,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> nearest_rows,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> covers,
    int32_t panel_size,
    int32_t maximum_target_edges,
    int32_t required_taxa_cap
) {
    if (edges.ndim() != 2 || edges.shape(1) != 2) {
        throw std::runtime_error("edges must have shape (undirected_edges, 2)");
    }
    if (nearest_rows.ndim() != 2 || nearest_rows.shape(1) < 3) {
        throw std::runtime_error("nearest_rows must have shape (directed_edges, 2 + limit)");
    }
    if (covers.ndim() != 1) {
        throw std::runtime_error("covers must be one-dimensional");
    }
    if (node_count <= 0 || taxon_count <= 0 || taxon_count > node_count ||
        panel_size <= 0 || panel_size > taxon_count ||
        maximum_target_edges <= 0 || required_taxa_cap < 4 ||
        required_taxa_cap > panel_size) {
        throw std::runtime_error("invalid contextual-cover dimensions");
    }

    const auto pairs = edges.unchecked<2>();
    const auto nearest_input = nearest_rows.unchecked<2>();
    const auto cover_values = covers.unchecked<1>();
    std::vector<std::vector<int32_t>> adjacency(node_count);
    for (py::ssize_t row = 0; row < edges.shape(0); ++row) {
        const int32_t left = pairs(row, 0);
        const int32_t right = pairs(row, 1);
        if (left < 0 || left >= node_count || right < 0 || right >= node_count ||
            left == right) {
            throw std::runtime_error("edge endpoint is outside the tree");
        }
        adjacency[left].push_back(right);
        adjacency[right].push_back(left);
    }
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        if (std::adjacent_find(neighbors.begin(), neighbors.end()) != neighbors.end()) {
            throw std::runtime_error("tree contains a duplicate edge");
        }
        if (neighbors.size() > 3) {
            throw std::runtime_error("contextual covers require a binary tree");
        }
    }
    if (edges.shape(0) != node_count - 1) {
        throw std::runtime_error("contextual covers require a tree");
    }

    const int32_t nearest_limit = static_cast<int32_t>(nearest_rows.shape(1) - 2);
    std::vector<std::array<std::vector<int32_t>, 3>> nearest(node_count);
    const auto nearest_at = [&](int32_t branch, int32_t owner)
        -> std::vector<int32_t>& {
        const auto found = std::lower_bound(
            adjacency[branch].begin(), adjacency[branch].end(), owner
        );
        if (found == adjacency[branch].end() || *found != owner) {
            throw std::runtime_error("nearest-message endpoints are not adjacent");
        }
        const auto slot = static_cast<size_t>(found - adjacency[branch].begin());
        return nearest[branch][slot];
    };
    for (py::ssize_t row = 0; row < nearest_rows.shape(0); ++row) {
        const int32_t branch = nearest_input(row, 0);
        const int32_t owner = nearest_input(row, 1);
        if (branch < 0 || branch >= node_count || owner < 0 || owner >= node_count) {
            throw std::runtime_error("nearest-message endpoint is outside the tree");
        }
        auto& values = nearest_at(branch, owner);
        for (int32_t column = 0; column < nearest_limit; ++column) {
            const int32_t taxon = nearest_input(row, column + 2);
            if (taxon >= 0) {
                if (taxon >= taxon_count) {
                    throw std::runtime_error("nearest-message taxon is outside the tree");
                }
                values.push_back(taxon);
            }
        }
    }

    std::vector<int32_t> parent(node_count, -2);
    std::vector<int32_t> depth(node_count, 0);
    std::vector<int32_t> order;
    order.reserve(node_count);
    parent[0] = 0;
    order.push_back(0);
    for (size_t offset = 0; offset < order.size(); ++offset) {
        const int32_t node = order[offset];
        for (const int32_t neighbor : adjacency[node]) {
            if (neighbor == parent[node]) {
                continue;
            }
            if (parent[neighbor] != -2) {
                throw std::runtime_error("adjacency is not a tree");
            }
            parent[neighbor] = node;
            depth[neighbor] = depth[node] + 1;
            order.push_back(neighbor);
        }
    }
    if (static_cast<int32_t>(order.size()) != node_count) {
        throw std::runtime_error("adjacency is disconnected");
    }
    int32_t levels = 1;
    while ((int64_t{1} << levels) <= node_count) {
        ++levels;
    }
    std::vector<std::vector<int32_t>> ancestors(
        levels, std::vector<int32_t>(node_count, 0)
    );
    ancestors[0] = parent;
    for (int32_t level = 1; level < levels; ++level) {
        for (int32_t node = 0; node < node_count; ++node) {
            ancestors[level][node] = ancestors[level - 1][ancestors[level - 1][node]];
        }
    }
    const auto distance = [&](int32_t left, int32_t right) {
        const int32_t original_left_depth = depth[left];
        const int32_t original_right_depth = depth[right];
        if (depth[left] < depth[right]) {
            std::swap(left, right);
        }
        int32_t difference = depth[left] - depth[right];
        for (int32_t level = 0; level < levels; ++level) {
            if ((difference >> level) & 1) {
                left = ancestors[level][left];
            }
        }
        if (left != right) {
            for (int32_t level = levels - 1; level >= 0; --level) {
                if (ancestors[level][left] != ancestors[level][right]) {
                    left = ancestors[level][left];
                    right = ancestors[level][right];
                }
            }
            left = ancestors[0][left];
        }
        return original_left_depth + original_right_depth - 2 * depth[left];
    };

    using EdgePair = std::array<int32_t, 2>;
    std::vector<EdgePair> internal;
    for (py::ssize_t row = 0; row < edges.shape(0); ++row) {
        int32_t left = pairs(row, 0);
        int32_t right = pairs(row, 1);
        if (left > right) {
            std::swap(left, right);
        }
        if (left >= taxon_count && right >= taxon_count) {
            internal.push_back({left, right});
        }
    }
    std::sort(internal.begin(), internal.end());
    if (std::adjacent_find(internal.begin(), internal.end()) != internal.end()) {
        throw std::runtime_error("duplicate internal edge");
    }
    const int32_t internal_count = static_cast<int32_t>(internal.size());
    std::vector<std::vector<int32_t>> incident(node_count);
    for (int32_t index = 0; index < internal_count; ++index) {
        incident[internal[index][0]].push_back(index);
        incident[internal[index][1]].push_back(index);
    }
    std::vector<std::vector<int32_t>> edge_neighbors(internal_count);
    std::vector<std::array<int32_t, 4>> branches(internal_count);
    for (int32_t index = 0; index < internal_count; ++index) {
        const auto edge = internal[index];
        auto candidates = incident[edge[0]];
        candidates.insert(candidates.end(), incident[edge[1]].begin(), incident[edge[1]].end());
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        candidates.erase(std::remove(candidates.begin(), candidates.end(), index), candidates.end());
        edge_neighbors[index] = std::move(candidates);
        std::array<int32_t, 4> values{};
        int offset = 0;
        for (const int32_t neighbor : adjacency[edge[0]]) {
            if (neighbor != edge[1]) values[offset++] = neighbor;
        }
        for (const int32_t neighbor : adjacency[edge[1]]) {
            if (neighbor != edge[0]) values[offset++] = neighbor;
        }
        if (offset != 4) {
            throw std::runtime_error("internal edge is not binary");
        }
        branches[index] = values;
    }

    struct NativePanel {
        int32_t cover;
        int32_t panel_id;
        std::vector<int32_t> taxa;
        std::vector<EdgePair> targets;
    };
    std::vector<NativePanel> panels;
    {
        py::gil_scoped_release release;
        for (py::ssize_t cover_offset = 0; cover_offset < covers.shape(0); ++cover_offset) {
            const int32_t cover = cover_values(cover_offset);
            const bool reverse = (cover % 2) != 0;
            const int32_t rank = ((cover % 4) + 4) % 4;
            std::vector<std::array<int32_t, 4>> representatives(internal_count);
            for (int32_t edge_index = 0; edge_index < internal_count; ++edge_index) {
                const auto edge = internal[edge_index];
                const int32_t owners[4] = {edge[0], edge[0], edge[1], edge[1]};
                for (int branch_offset = 0; branch_offset < 4; ++branch_offset) {
                    const auto& values = nearest_at(
                        branches[edge_index][branch_offset], owners[branch_offset]
                    );
                    if (values.empty()) {
                        throw std::runtime_error("directed nearest message is missing");
                    }
                    representatives[edge_index][branch_offset] =
                        values[std::min<int32_t>(rank, static_cast<int32_t>(values.size()) - 1)];
                }
                auto unique = representatives[edge_index];
                std::sort(unique.begin(), unique.end());
                if (std::adjacent_find(unique.begin(), unique.end()) != unique.end()) {
                    throw std::runtime_error("edge representatives are not distinct");
                }
            }

            std::vector<uint8_t> remaining(internal_count, 1);
            int32_t remaining_count = internal_count;
            int32_t seed_position = reverse ? internal_count - 1 : 0;
            int32_t panel_id = 0;
            while (remaining_count > 0) {
                while (seed_position >= 0 && seed_position < internal_count &&
                       !remaining[seed_position]) {
                    seed_position += reverse ? -1 : 1;
                }
                if (seed_position < 0 || seed_position >= internal_count) {
                    throw std::runtime_error("failed to find contextual-cover seed");
                }
                std::deque<int32_t> queue{seed_position};
                std::vector<uint8_t> queued(internal_count, 0);
                queued[seed_position] = 1;
                std::vector<int32_t> targets;
                std::vector<uint8_t> required(taxon_count, 0);
                int32_t required_count = 0;
                while (!queue.empty() &&
                       static_cast<int32_t>(targets.size()) < maximum_target_edges) {
                    const int32_t edge_index = queue.front();
                    queue.pop_front();
                    if (!remaining[edge_index]) continue;
                    int32_t proposed_count = required_count;
                    for (const int32_t taxon : representatives[edge_index]) {
                        proposed_count += !required[taxon];
                    }
                    if (!targets.empty() && proposed_count > required_taxa_cap) continue;
                    targets.push_back(edge_index);
                    for (const int32_t taxon : representatives[edge_index]) {
                        if (!required[taxon]) {
                            required[taxon] = 1;
                            ++required_count;
                        }
                    }
                    remaining[edge_index] = 0;
                    --remaining_count;
                    const auto& neighbors = edge_neighbors[edge_index];
                    if (reverse) {
                        for (auto iterator = neighbors.rbegin(); iterator != neighbors.rend(); ++iterator) {
                            if (remaining[*iterator] && !queued[*iterator]) {
                                queued[*iterator] = 1;
                                queue.push_back(*iterator);
                            }
                        }
                    } else {
                        for (const int32_t neighbor : neighbors) {
                            if (remaining[neighbor] && !queued[neighbor]) {
                                queued[neighbor] = 1;
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
                if (targets.empty()) {
                    throw std::runtime_error("failed to cover contextual seed edge");
                }

                std::vector<uint8_t> selected = required;
                int32_t selected_count = required_count;
                for (const int32_t edge_index : targets) {
                    const auto edge = internal[edge_index];
                    const int32_t owners[4] = {edge[0], edge[0], edge[1], edge[1]};
                    for (int branch_offset = 0; branch_offset < 4; ++branch_offset) {
                        const auto& values = nearest_at(
                            branches[edge_index][branch_offset], owners[branch_offset]
                        );
                        if (reverse) {
                            for (auto iterator = values.rbegin(); iterator != values.rend(); ++iterator) {
                                if (selected_count == panel_size) break;
                                if (!selected[*iterator]) {
                                    selected[*iterator] = 1;
                                    ++selected_count;
                                }
                            }
                        } else {
                            for (const int32_t taxon : values) {
                                if (selected_count == panel_size) break;
                                if (!selected[taxon]) {
                                    selected[taxon] = 1;
                                    ++selected_count;
                                }
                            }
                        }
                    }
                    if (selected_count == panel_size) break;
                }

                if (selected_count < panel_size) {
                    std::vector<uint8_t> start_seen(node_count, 0);
                    std::vector<int32_t> starts;
                    for (const int32_t edge_index : targets) {
                        for (const int32_t node : internal[edge_index]) {
                            if (!start_seen[node]) {
                                start_seen[node] = 1;
                                starts.push_back(node);
                            }
                        }
                    }
                    std::sort(starts.begin(), starts.end());
                    if (reverse) std::reverse(starts.begin(), starts.end());
                    std::vector<uint8_t> candidate_seen(taxon_count, 0);
                    for (const int32_t node : starts) {
                        for (const int32_t neighbor : adjacency[node]) {
                            const auto& values = nearest_at(neighbor, node);
                            for (const int32_t taxon : values) {
                                if (!selected[taxon]) candidate_seen[taxon] = 1;
                            }
                        }
                    }
                    std::vector<std::pair<int32_t, int32_t>> ranked;
                    for (int32_t taxon = 0; taxon < taxon_count; ++taxon) {
                        if (!candidate_seen[taxon]) continue;
                        int32_t nearest_distance = std::numeric_limits<int32_t>::max();
                        for (const int32_t node : starts) {
                            nearest_distance = std::min(nearest_distance, distance(node, taxon));
                        }
                        ranked.emplace_back(nearest_distance, taxon);
                    }
                    std::sort(ranked.begin(), ranked.end(), [&](const auto& left, const auto& right) {
                        if (left.first != right.first) return left.first < right.first;
                        return reverse ? left.second > right.second : left.second < right.second;
                    });
                    for (const auto& value : ranked) {
                        if (selected_count == panel_size) break;
                        if (!selected[value.second]) {
                            selected[value.second] = 1;
                            ++selected_count;
                        }
                    }
                }
                if (selected_count < panel_size) {
                    if (reverse) {
                        for (int32_t taxon = taxon_count - 1; taxon >= 0 && selected_count < panel_size; --taxon) {
                            if (!selected[taxon]) {
                                selected[taxon] = 1;
                                ++selected_count;
                            }
                        }
                    } else {
                        for (int32_t taxon = 0; taxon < taxon_count && selected_count < panel_size; ++taxon) {
                            if (!selected[taxon]) {
                                selected[taxon] = 1;
                                ++selected_count;
                            }
                        }
                    }
                }
                if (selected_count != panel_size) {
                    throw std::runtime_error("native contextual panel has wrong size");
                }
                NativePanel panel{cover, panel_id++, {}, {}};
                panel.taxa.reserve(panel_size);
                for (int32_t taxon = 0; taxon < taxon_count; ++taxon) {
                    if (selected[taxon]) panel.taxa.push_back(taxon);
                }
                panel.targets.reserve(targets.size());
                for (const int32_t edge_index : targets) {
                    panel.targets.push_back(internal[edge_index]);
                }
                panels.push_back(std::move(panel));
            }
        }
    }

    py::list output;
    for (const auto& panel : panels) {
        py::array_t<int32_t> taxa(panel.taxa.size());
        std::copy(panel.taxa.begin(), panel.taxa.end(), taxa.mutable_data());
        py::array_t<int32_t> targets({static_cast<py::ssize_t>(panel.targets.size()), py::ssize_t{2}});
        auto target_rows = targets.mutable_unchecked<2>();
        for (py::ssize_t row = 0; row < targets.shape(0); ++row) {
            target_rows(row, 0) = panel.targets[row][0];
            target_rows(row, 1) = panel.targets[row][1];
        }
        output.append(py::make_tuple(panel.cover, panel.panel_id, taxa, targets));
    }
    return output;
}

py::tuple median_pair(
    py::array_t<double, py::array::c_style | py::array::forcecast> first,
    py::array_t<double, py::array::c_style | py::array::forcecast> second
) {
    if (first.ndim() != 1 || second.ndim() != 1) {
        throw std::runtime_error("median inputs must be one-dimensional");
    }
    double first_median;
    double second_median;
    {
        py::gil_scoped_release release;
        first_median = median_copy(first.data(), first.size());
        second_median = median_copy(second.data(), second.size());
    }
    return py::make_tuple(first_median, second_median);
}

py::array_t<int8_t> compile_panel_current_classes(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> panel_positions,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> depth,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> ancestors,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> quartets
) {
    if (panel_positions.ndim() != 1 || depth.ndim() != 1) {
        throw std::runtime_error("panel_positions and depth must be one-dimensional");
    }
    if (ancestors.ndim() != 2 || ancestors.shape(1) != depth.shape(0)) {
        throw std::runtime_error("ancestors must have shape (levels, nodes)");
    }
    if (quartets.ndim() != 2 || quartets.shape(1) != 4) {
        throw std::runtime_error("quartets must have shape (rows, 4)");
    }
    const auto positions = panel_positions.unchecked<1>();
    const auto node_depth = depth.unchecked<1>();
    const auto jump = ancestors.unchecked<2>();
    const auto rows = quartets.unchecked<2>();
    const py::ssize_t panel_size = panel_positions.shape(0);
    for (py::ssize_t index = 0; index < panel_size; ++index) {
        if (positions(index) < 0 || positions(index) >= depth.shape(0)) {
            throw std::runtime_error("panel position is outside the tree index");
        }
    }

    std::vector<int32_t> distances(panel_size * panel_size, 0);
    for (py::ssize_t left = 0; left < panel_size; ++left) {
        for (py::ssize_t right = left + 1; right < panel_size; ++right) {
            const int32_t distance = tree_distance(
                positions(left), positions(right), node_depth, jump
            );
            distances[left * panel_size + right] = distance;
            distances[right * panel_size + left] = distance;
        }
    }

    py::array_t<int8_t> output(quartets.shape(0));
    auto classes = output.mutable_unchecked<1>();
    for (py::ssize_t row = 0; row < quartets.shape(0); ++row) {
        int16_t local[4];
        for (int column = 0; column < 4; ++column) {
            local[column] = rows(row, column);
            if (local[column] < 0 || local[column] >= panel_size) {
                throw std::runtime_error("quartet position is outside the panel");
            }
        }
        const auto distance = [&](int left, int right) {
            return distances[local[left] * panel_size + local[right]];
        };
        const int32_t sums[3] = {
            distance(0, 1) + distance(2, 3),
            distance(0, 2) + distance(1, 3),
            distance(0, 3) + distance(1, 2),
        };
        int8_t best = 0;
        if (sums[1] < sums[best]) {
            best = 1;
        }
        if (sums[2] < sums[best]) {
            best = 2;
        }
        classes(row) = best;
    }
    return output;
}

py::list compile_panel_edge_rows(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> panel_entered,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> descriptors,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> row_lookup
) {
    if (panel_entered.ndim() != 1) {
        throw std::runtime_error("panel_entered must have shape (panel_size,)");
    }
    const auto panel_size = panel_entered.shape(0);
    if (descriptors.ndim() != 3 || descriptors.shape(1) != 4 ||
        descriptors.shape(2) != 3) {
        throw std::runtime_error("descriptors must have shape (edges, 4, 3)");
    }
    if (row_lookup.ndim() != 4 || row_lookup.shape(0) != panel_size ||
        row_lookup.shape(1) != panel_size || row_lookup.shape(2) != panel_size ||
        row_lookup.shape(3) != panel_size) {
        throw std::runtime_error("row_lookup must have shape (panel_size,)*4");
    }

    const auto ticks = panel_entered.unchecked<1>();
    const auto sides = descriptors.unchecked<3>();
    const auto lookup = row_lookup.unchecked<4>();
    py::list output;

    for (py::ssize_t edge = 0; edge < descriptors.shape(0); ++edge) {
        std::array<std::vector<int16_t>, 4> groups;
        for (py::ssize_t position = 0; position < panel_size; ++position) {
            int matched = -1;
            for (int group = 0; group < 4; ++group) {
                const bool complement = sides(edge, group, 2) != 0;
                if (in_component(
                        ticks(position), sides(edge, group, 0),
                        sides(edge, group, 1), complement)) {
                    if (matched != -1) {
                        throw std::runtime_error("directed components overlap");
                    }
                    matched = group;
                }
            }
            if (matched == -1) {
                throw std::runtime_error("panel taxon has no directed component");
            }
            groups[matched].push_back(static_cast<int16_t>(position));
        }
        for (const auto& group : groups) {
            if (group.empty()) {
                throw std::runtime_error("panel does not distinguish target edge");
            }
        }

        const py::ssize_t count =
            static_cast<py::ssize_t>(groups[0].size()) *
            static_cast<py::ssize_t>(groups[1].size()) *
            static_cast<py::ssize_t>(groups[2].size()) *
            static_cast<py::ssize_t>(groups[3].size());
        py::array_t<int64_t> changed(count);
        py::array_t<int8_t> current(count);
        py::array_t<int8_t> alt1(count);
        py::array_t<int8_t> alt2(count);
        auto changed_out = changed.mutable_unchecked<1>();
        auto current_out = current.mutable_unchecked<1>();
        auto alt1_out = alt1.mutable_unchecked<1>();
        auto alt2_out = alt2.mutable_unchecked<1>();

        py::ssize_t offset = 0;
        for (const int one : groups[0]) {
            for (const int two : groups[1]) {
                for (const int three : groups[2]) {
                    for (const int four : groups[3]) {
                        const int32_t row = lookup(one, two, three, four);
                        if (row < 0) {
                            throw std::runtime_error("dense quartet row is missing");
                        }
                        const int values[4] = {one, two, three, four};
                        int ranks[4] = {0, 0, 0, 0};
                        for (int left = 0; left < 4; ++left) {
                            for (int right = left + 1; right < 4; ++right) {
                                if (values[left] > values[right]) {
                                    ++ranks[left];
                                } else {
                                    ++ranks[right];
                                }
                            }
                        }
                        changed_out(offset) = static_cast<int64_t>(row);
                        current_out(offset) = pair_class(ranks[0], ranks[1]);
                        alt1_out(offset) = pair_class(ranks[0], ranks[2]);
                        alt2_out(offset) = pair_class(ranks[0], ranks[3]);
                        ++offset;
                    }
                }
            }
        }
        if (offset != count) {
            throw std::runtime_error("compiled row count mismatch");
        }
        output.append(py::make_tuple(changed, current, alt1, alt2));
    }
    return output;
}

py::array_t<double> score_sparse_panel_sums(
    py::array_t<float, py::array::c_style | py::array::forcecast> probabilities,
    py::list positions_list,
    py::list current_list,
    py::list alternative_one_list,
    py::list alternative_two_list,
    int64_t emitted_rows
) {
    if (probabilities.ndim() != 2 || probabilities.shape(1) != 3 ||
        emitted_rows <= 0) {
        throw std::runtime_error("invalid sparse probabilities or emitted row count");
    }
    const py::ssize_t edge_count = positions_list.size();
    if (current_list.size() != edge_count ||
        alternative_one_list.size() != edge_count ||
        alternative_two_list.size() != edge_count) {
        throw std::runtime_error("sparse edge arrays have inconsistent counts");
    }
    const auto values = probabilities.unchecked<2>();
    std::vector<double> logp(static_cast<size_t>(probabilities.shape(0) * 3));
    for (py::ssize_t row = 0; row < probabilities.shape(0); ++row) {
        for (int state = 0; state < 3; ++state) {
            const double value = std::min(
                1.0, std::max(1e-8, static_cast<double>(values(row, state)))
            );
            logp[static_cast<size_t>(row * 3 + state)] = std::log(value);
        }
    }
    py::array_t<double> output({edge_count, py::ssize_t{5}});
    auto result = output.mutable_unchecked<2>();
    for (py::ssize_t edge = 0; edge < edge_count; ++edge) {
        auto positions = py::cast<py::array_t<int64_t,
            py::array::c_style | py::array::forcecast>>(positions_list[edge]);
        auto current = py::cast<py::array_t<int8_t,
            py::array::c_style | py::array::forcecast>>(current_list[edge]);
        auto alternative_one = py::cast<py::array_t<int8_t,
            py::array::c_style | py::array::forcecast>>(alternative_one_list[edge]);
        auto alternative_two = py::cast<py::array_t<int8_t,
            py::array::c_style | py::array::forcecast>>(alternative_two_list[edge]);
        if (positions.ndim() != 1 || current.ndim() != 1 ||
            alternative_one.ndim() != 1 || alternative_two.ndim() != 1 ||
            current.shape(0) != positions.shape(0) ||
            alternative_one.shape(0) != positions.shape(0) ||
            alternative_two.shape(0) != positions.shape(0) ||
            positions.shape(0) == 0) {
            throw std::runtime_error("invalid sparse edge arrays");
        }
        const auto position_values = positions.unchecked<1>();
        const auto current_values = current.unchecked<1>();
        const auto first_values = alternative_one.unchecked<1>();
        const auto second_values = alternative_two.unchecked<1>();
        double first_sum = 0.0;
        double second_sum = 0.0;
        for (py::ssize_t row = 0; row < positions.shape(0); ++row) {
            const int64_t position = position_values(row);
            const int one = current_values(row);
            const int two = first_values(row);
            const int three = second_values(row);
            if (position < 0 || position >= probabilities.shape(0) ||
                one < 0 || one >= 3 || two < 0 || two >= 3 ||
                three < 0 || three >= 3) {
                throw std::runtime_error("sparse edge index is out of range");
            }
            const size_t offset = static_cast<size_t>(position * 3);
            first_sum += logp[offset + two] - logp[offset + one];
            second_sum += logp[offset + three] - logp[offset + one];
        }
        const double first_score = first_sum / static_cast<double>(emitted_rows);
        const double second_score = second_sum / static_cast<double>(emitted_rows);
        int best = 0;
        double gain = 0.0;
        if (first_score > gain) {
            best = 1;
            gain = first_score;
        }
        if (second_score > gain) {
            best = 2;
            gain = second_score;
        }
        result(edge, 0) = first_score;
        result(edge, 1) = second_score;
        result(edge, 2) = static_cast<double>(best);
        result(edge, 3) = gain;
        result(edge, 4) = static_cast<double>(positions.shape(0));
    }
    return output;
}

PYBIND11_MODULE(panel_score_backend, module) {
    module.def(
        "compile_directed_nearest_messages",
        &compile_directed_nearest_messages,
        "Compile exact nearest-leaf tuples for every directed tree edge"
    );
    module.def(
        "compile_contextual_panel_covers",
        &compile_contextual_panel_covers,
        "Compile exact deterministic connected contextual-panel covers"
    );
    module.def(
        "median_pair",
        &median_pair,
        "Return exact ordinary medians for two finite float64 vectors"
    );
    module.def(
        "compile_panel_current_classes",
        &compile_panel_current_classes,
        "Classify displayed quartets from one exact panel distance table"
    );
    module.def(
        "compile_panel_edge_rows",
        &compile_panel_edge_rows,
        "Compile exact ordered integer panel edge plans"
    );
    module.def(
        "score_sparse_panel_sums",
        &score_sparse_panel_sums,
        "Reduce sparse independent-row edge likelihood sums"
    );
}
