#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

bool row_subset(
    const uint64_t* left,
    const uint64_t* right,
    py::ssize_t width
) {
    for (py::ssize_t word = 0; word < width; ++word) {
        if ((left[word] & right[word]) != left[word]) {
            return false;
        }
    }
    return true;
}

class LaminarHierarchy {
public:
    LaminarHierarchy(py::ssize_t taxa, py::ssize_t width, py::ssize_t capacity)
        : taxa_(taxa), width_(width), root_(taxa + capacity),
          parent_(static_cast<std::size_t>(root_ + 1), root_),
          marks_(static_cast<std::size_t>(root_ + 1), 0) {
        if (taxa_ < 4 || width_ != (taxa_ + 63) / 64) {
            throw std::runtime_error("invalid taxon count for packed split width");
        }
        parent_[static_cast<std::size_t>(root_)] = root_;
        masks_.reserve(static_cast<std::size_t>(capacity) * static_cast<std::size_t>(width_));
        top_children_.reserve(static_cast<std::size_t>(taxa_));
    }

    bool insert(const uint64_t* candidate) {
        const py::ssize_t first_leaf = first_set_leaf(candidate);
        if (first_leaf < 0 || first_leaf >= taxa_) {
            return false;
        }
        py::ssize_t ancestor = parent_[static_cast<std::size_t>(first_leaf)];
        while (ancestor != root_ && !row_subset(candidate, mask(ancestor), width_)) {
            ancestor = parent_[static_cast<std::size_t>(ancestor)];
        }

        top_children_.clear();
        if (++stamp_ == 0) {
            std::fill(marks_.begin(), marks_.end(), 0);
            stamp_ = 1;
        }
        for (py::ssize_t word = 0; word < width_; ++word) {
            uint64_t remaining = candidate[word];
            while (remaining != 0) {
                const unsigned bit = static_cast<unsigned>(__builtin_ctzll(remaining));
                const py::ssize_t leaf = word * 64 + static_cast<py::ssize_t>(bit);
                remaining &= remaining - 1;
                if (leaf >= taxa_) {
                    return false;
                }
                py::ssize_t child = leaf;
                while (parent_[static_cast<std::size_t>(child)] != ancestor) {
                    child = parent_[static_cast<std::size_t>(child)];
                    if (child == root_) {
                        throw std::runtime_error("laminar hierarchy parent chain is inconsistent");
                    }
                }
                if (child >= taxa_ && !row_subset(mask(child), candidate, width_)) {
                    return false;
                }
                if (marks_[static_cast<std::size_t>(child)] != stamp_) {
                    marks_[static_cast<std::size_t>(child)] = stamp_;
                    top_children_.push_back(child);
                }
            }
        }

        const py::ssize_t node = taxa_ + static_cast<py::ssize_t>(masks_.size()) / width_;
        if (node >= root_) {
            throw std::runtime_error("laminar hierarchy capacity exceeded");
        }
        masks_.insert(masks_.end(), candidate, candidate + width_);
        parent_[static_cast<std::size_t>(node)] = ancestor;
        for (const py::ssize_t child : top_children_) {
            parent_[static_cast<std::size_t>(child)] = node;
        }
        return true;
    }

private:
    const uint64_t* mask(py::ssize_t node) const {
        if (node < taxa_ || node >= root_) {
            throw std::runtime_error("requested mask for a non-cluster hierarchy node");
        }
        return masks_.data() + static_cast<std::size_t>(node - taxa_) * static_cast<std::size_t>(width_);
    }

    py::ssize_t first_set_leaf(const uint64_t* row) const {
        for (py::ssize_t word = 0; word < width_; ++word) {
            if (row[word] != 0) {
                return word * 64 + static_cast<py::ssize_t>(__builtin_ctzll(row[word]));
            }
        }
        return -1;
    }

    py::ssize_t taxa_;
    py::ssize_t width_;
    py::ssize_t root_;
    std::vector<py::ssize_t> parent_;
    std::vector<uint64_t> masks_;
    std::vector<uint32_t> marks_;
    uint32_t stamp_ = 0;
    std::vector<py::ssize_t> top_children_;
};

}  // namespace

py::array_t<uint8_t> greedy_laminar_accept_hierarchy(
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> accepted,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> candidates,
    py::ssize_t maximum_accepts,
    py::ssize_t n_taxa
) {
    if (accepted.ndim() != 2 || candidates.ndim() != 2) {
        throw std::runtime_error("accepted and candidates must be two-dimensional");
    }
    if (accepted.shape(1) != candidates.shape(1)) {
        throw std::runtime_error("accepted and candidates must have equal word width");
    }
    if (maximum_accepts < 0) {
        throw std::runtime_error("maximum_accepts must be nonnegative");
    }
    const py::ssize_t width = candidates.shape(1);
    const py::ssize_t initial_count = accepted.shape(0);
    const py::ssize_t candidate_count = candidates.shape(0);
    const auto accepted_in = accepted.unchecked<2>();
    const auto candidate_in = candidates.unchecked<2>();
    py::array_t<uint8_t> output(candidate_count);
    auto keep = output.mutable_unchecked<1>();
    std::fill_n(keep.mutable_data(0), candidate_count, uint8_t{0});

    py::gil_scoped_release release;
    LaminarHierarchy hierarchy(
        n_taxa, width, initial_count + std::min(candidate_count, maximum_accepts)
    );
    for (py::ssize_t row = 0; row < initial_count; ++row) {
        if (!hierarchy.insert(&accepted_in(row, 0))) {
            throw std::runtime_error("accepted split prefix is not laminar");
        }
    }
    py::ssize_t newly_accepted = 0;
    for (py::ssize_t row = 0; row < candidate_count; ++row) {
        if (newly_accepted >= maximum_accepts) {
            break;
        }
        if (hierarchy.insert(&candidate_in(row, 0))) {
            keep(row) = uint8_t{1};
            ++newly_accepted;
        }
    }
    return output;
}

py::array_t<uint8_t> greedy_laminar_accept(
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> accepted,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> candidates,
    py::ssize_t maximum_accepts
) {
    if (accepted.ndim() != 2 || candidates.ndim() != 2) {
        throw std::runtime_error("accepted and candidates must be two-dimensional");
    }
    if (accepted.shape(1) != candidates.shape(1)) {
        throw std::runtime_error("accepted and candidates must have equal word width");
    }
    if (maximum_accepts < 0) {
        throw std::runtime_error("maximum_accepts must be nonnegative");
    }

    const py::ssize_t width = candidates.shape(1);
    const py::ssize_t initial_count = accepted.shape(0);
    const py::ssize_t candidate_count = candidates.shape(0);
    const auto accepted_in = accepted.unchecked<2>();
    const auto candidate_in = candidates.unchecked<2>();
    py::array_t<uint8_t> output(candidate_count);
    auto keep = output.mutable_unchecked<1>();
    std::fill_n(keep.mutable_data(0), candidate_count, uint8_t{0});

    std::vector<uint64_t> selected;
    selected.reserve(
        static_cast<std::size_t>(initial_count + maximum_accepts) *
        static_cast<std::size_t>(width)
    );
    for (py::ssize_t row = 0; row < initial_count; ++row) {
        for (py::ssize_t word = 0; word < width; ++word) {
            selected.push_back(accepted_in(row, word));
        }
    }

    py::gil_scoped_release release;
    py::ssize_t selected_count = initial_count;
    py::ssize_t newly_accepted = 0;
    for (py::ssize_t row = 0; row < candidate_count; ++row) {
        if (newly_accepted >= maximum_accepts) {
            break;
        }
        bool compatible_with_all = true;
        for (py::ssize_t old = 0; old < selected_count; ++old) {
            bool disjoint = true;
            bool candidate_subset = true;
            bool old_subset = true;
            const auto base = static_cast<std::size_t>(old * width);
            for (py::ssize_t word = 0; word < width; ++word) {
                const uint64_t candidate_word = candidate_in(row, word);
                const uint64_t old_word = selected[base + word];
                const uint64_t intersection = candidate_word & old_word;
                disjoint = disjoint && intersection == 0;
                candidate_subset = candidate_subset && intersection == candidate_word;
                old_subset = old_subset && intersection == old_word;
            }
            if (!(disjoint || candidate_subset || old_subset)) {
                compatible_with_all = false;
                break;
            }
        }
        if (!compatible_with_all) {
            continue;
        }
        keep(row) = uint8_t{1};
        for (py::ssize_t word = 0; word < width; ++word) {
            selected.push_back(candidate_in(row, word));
        }
        ++selected_count;
        ++newly_accepted;
    }
    return output;
}

py::tuple laminar_parent_indices(
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> clusters,
    py::ssize_t n_taxa
) {
    if (clusters.ndim() != 2 || n_taxa < 4 ||
        clusters.shape(1) != (n_taxa + 63) / 64) {
        throw std::runtime_error("invalid packed laminar cluster matrix");
    }
    const py::ssize_t count = clusters.shape(0);
    const py::ssize_t width = clusters.shape(1);
    const py::ssize_t root = count;
    const uint64_t* raw = clusters.data();
    py::array_t<int64_t> cluster_parent(count);
    py::array_t<int64_t> leaf_owner(n_taxa);
    int64_t* parents = cluster_parent.mutable_data();
    int64_t* owners = leaf_owner.mutable_data();
    std::fill_n(owners, n_taxa, static_cast<int64_t>(root));

    {
        py::gil_scoped_release release;
        for (py::ssize_t row = 0; row < count; ++row) {
            const uint64_t* candidate = raw + row * width;
            py::ssize_t representative = -1;
            for (py::ssize_t word = 0; word < width; ++word) {
                if (candidate[word] != 0) {
                    representative = word * 64 + static_cast<py::ssize_t>(
                        __builtin_ctzll(candidate[word])
                    );
                    break;
                }
            }
            if (representative < 0 || representative >= n_taxa) {
                throw std::runtime_error("laminar cluster is empty or out of range");
            }
            const int64_t parent = owners[representative];
            if (parent != root &&
                !row_subset(candidate, raw + parent * width, width)) {
                throw std::runtime_error("oriented split clusters are not laminar");
            }
            parents[row] = parent;
            for (py::ssize_t word = 0; word < width; ++word) {
                uint64_t remaining = candidate[word];
                while (remaining != 0) {
                    const unsigned bit = static_cast<unsigned>(__builtin_ctzll(remaining));
                    const py::ssize_t leaf = word * 64 + static_cast<py::ssize_t>(bit);
                    remaining &= remaining - 1;
                    if (leaf >= n_taxa || owners[leaf] != parent) {
                        throw std::runtime_error("oriented split clusters are not laminar");
                    }
                    owners[leaf] = static_cast<int64_t>(row);
                }
            }
        }
    }
    return py::make_tuple(cluster_parent, leaf_owner);
}

PYBIND11_MODULE(split_compat_backend, module) {
    module.doc() = "Exact packed greedy anchored-laminar split selector";
    module.def(
        "greedy_laminar_accept",
        &greedy_laminar_accept,
        py::arg("accepted"),
        py::arg("candidates"),
        py::arg("maximum_accepts")
    );
    module.def(
        "greedy_laminar_accept_hierarchy",
        &greedy_laminar_accept_hierarchy,
        py::arg("accepted"),
        py::arg("candidates"),
        py::arg("maximum_accepts"),
        py::arg("n_taxa")
    );
    module.def(
        "laminar_parent_indices",
        &laminar_parent_indices,
        py::arg("clusters"),
        py::arg("n_taxa")
    );
}
