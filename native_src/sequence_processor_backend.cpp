#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {
constexpr uint8_t kGapState = 4;
constexpr uint8_t kPackedGapByte = static_cast<uint8_t>(kGapState | (kGapState << 4));

uint8_t map_base_to_state(char c) {
    static uint8_t lookup[256];
    static bool initialized = false;
    if (!initialized) {
        std::fill(std::begin(lookup), std::end(lookup), kGapState);
        lookup[static_cast<unsigned char>('A')] = lookup[static_cast<unsigned char>('a')] = 0;
        lookup[static_cast<unsigned char>('C')] = lookup[static_cast<unsigned char>('c')] = 1;
        lookup[static_cast<unsigned char>('G')] = lookup[static_cast<unsigned char>('g')] = 2;
        lookup[static_cast<unsigned char>('T')] = lookup[static_cast<unsigned char>('t')] = 3;
        lookup[static_cast<unsigned char>('U')] = lookup[static_cast<unsigned char>('u')] = 3;
        lookup[static_cast<unsigned char>('-')] = kGapState;
        initialized = true;
    }
    return lookup[static_cast<unsigned char>(c)];
}

inline uint8_t unpack_state(const uint8_t* row, int pos) {
    const uint8_t byte = row[pos >> 1];
    return static_cast<uint8_t>((pos & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F));
}

long long parse_species_index(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && !std::isdigit(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
    if (i == s.size()) {
        throw std::runtime_error("Failed to parse species index from: " + s);
    }

    size_t j = i;
    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
        ++j;
    }
    return std::stoll(s.substr(i, j - i));
}

void reorder_if_numbered(
    std::vector<uint8_t>& packed_buffer,
    std::vector<std::string>& species_names,
    int n_species,
    int packed_len
) {
    bool can_reorder = true;
    std::vector<uint8_t> used(static_cast<size_t>(n_species), 0);
    std::vector<std::string> ordered_names(static_cast<size_t>(n_species));
    std::vector<uint8_t> ordered_buffer(
        static_cast<size_t>(n_species) * static_cast<size_t>(packed_len),
        kPackedGapByte
    );

    for (int r = 0; r < n_species; ++r) {
        long long idx_ll = -1;
        try {
            idx_ll = parse_species_index(species_names[static_cast<size_t>(r)]);
        } catch (...) {
            can_reorder = false;
            break;
        }
        if (idx_ll < 0 || idx_ll >= n_species) {
            can_reorder = false;
            break;
        }

        const size_t idx = static_cast<size_t>(idx_ll);
        if (used[idx]) {
            can_reorder = false;
            break;
        }
        used[idx] = 1;
        ordered_names[idx] = species_names[static_cast<size_t>(r)];

        const uint8_t* src = packed_buffer.data() + static_cast<size_t>(r) * static_cast<size_t>(packed_len);
        uint8_t* dst = ordered_buffer.data() + idx * static_cast<size_t>(packed_len);
        std::copy(src, src + static_cast<size_t>(packed_len), dst);
    }

    if (can_reorder) {
        packed_buffer.swap(ordered_buffer);
        species_names.swap(ordered_names);
    }
}

template <typename Function>
void joined_parallel_for(std::size_t count, int workers, Function function) {
    const std::size_t thread_count = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(workers, 1)), count
    );
    if (thread_count <= 1) {
        for (std::size_t index = 0; index < count; ++index) {
            function(index);
        }
        return;
    }
    const std::size_t chunk = (count + thread_count - 1) / thread_count;
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        const std::size_t begin = thread * chunk;
        const std::size_t end = std::min(count, begin + chunk);
        if (begin >= end) {
            break;
        }
        threads.emplace_back([=, &function]() {
            for (std::size_t index = begin; index < end; ++index) {
                function(index);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}
}  // namespace

py::tuple load_phy_to_packed_tensor(const std::string& phy_path, bool drop_conserved_sites = false) {
    std::ifstream file(phy_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + phy_path);
    }

    std::string header;
    if (!std::getline(file, header)) {
        throw std::runtime_error("Failed to read header");
    }
    std::istringstream iss(header);
    int n_species = 0;
    int sequence_length = 0;
    if (!(iss >> n_species >> sequence_length) || n_species <= 0 || sequence_length <= 0) {
        throw std::runtime_error("Invalid header format");
    }

    const int packed_len = (sequence_length + 1) / 2;
    std::vector<uint8_t> packed_buffer(
        static_cast<size_t>(n_species) * static_cast<size_t>(packed_len),
        kPackedGapByte
    );
    std::vector<std::string> species_names;
    species_names.reserve(static_cast<size_t>(n_species));

    const size_t chunk_size = 1024 * 1024;
    std::vector<char> chunk_buffer(chunk_size);

    int species_idx = 0;
    std::string species_name;
    uint8_t* current_row = nullptr;
    int pos = 0;
    bool reading_name = true;

    while (species_idx < n_species && !file.eof()) {
        file.read(chunk_buffer.data(), static_cast<std::streamsize>(chunk_size));
        const std::streamsize bytes_read = file.gcount();

        for (std::streamsize i = 0; i < bytes_read; ++i) {
            const char c = chunk_buffer[static_cast<size_t>(i)];

            if (c == '\n') {
                if (!species_name.empty() && !reading_name) {
                    species_names.push_back(species_name);
                    ++species_idx;
                }
                species_name.clear();
                reading_name = true;
                current_row = nullptr;
                pos = 0;
                continue;
            }
            if (c == '\r') {
                continue;
            }

            if (reading_name) {
                if (c == ' ' || c == '\t') {
                    if (!species_name.empty()) {
                        reading_name = false;
                        current_row = packed_buffer.data() +
                                      static_cast<size_t>(species_idx) * static_cast<size_t>(packed_len);
                        pos = 0;
                    }
                } else {
                    species_name.push_back(c);
                }
            } else {
                if (c == ' ' || c == '\t') {
                    continue;
                }
                if (!current_row || pos >= sequence_length) {
                    continue;
                }

                const uint8_t state = map_base_to_state(c);
                const int packed_idx = pos >> 1;
                if (pos & 1) {
                    current_row[packed_idx] =
                        static_cast<uint8_t>((current_row[packed_idx] & 0x0F) | (state << 4));
                } else {
                    current_row[packed_idx] =
                        static_cast<uint8_t>((current_row[packed_idx] & 0xF0) | state);
                }
                ++pos;
            }
        }
    }

    if (species_idx < n_species && !species_name.empty() && !reading_name) {
        species_names.push_back(species_name);
        ++species_idx;
    }

    if (species_idx != n_species) {
        throw std::runtime_error(
            "PHY file ended early: expected " + std::to_string(n_species) +
            " species, got " + std::to_string(species_idx)
        );
    }

    reorder_if_numbered(packed_buffer, species_names, n_species, packed_len);

    int effective_sequence_length = sequence_length;
    if (drop_conserved_sites) {
        std::vector<int> kept_positions;
        kept_positions.reserve(static_cast<size_t>(sequence_length));

        for (int pos_idx = 0; pos_idx < sequence_length; ++pos_idx) {
            const uint8_t first_val = unpack_state(packed_buffer.data(), pos_idx);
            bool all_same = true;
            for (int sp = 1; sp < n_species; ++sp) {
                const uint8_t* row = packed_buffer.data() +
                                     static_cast<size_t>(sp) * static_cast<size_t>(packed_len);
                if (unpack_state(row, pos_idx) != first_val) {
                    all_same = false;
                    break;
                }
            }
            if (!all_same) {
                kept_positions.push_back(pos_idx);
            }
        }

        effective_sequence_length = static_cast<int>(kept_positions.size());
        const int filtered_packed_len = (effective_sequence_length + 1) / 2;
        std::vector<uint8_t> filtered_buffer(
            static_cast<size_t>(n_species) * static_cast<size_t>(filtered_packed_len),
            kPackedGapByte
        );

        for (int sp = 0; sp < n_species; ++sp) {
            const uint8_t* src_row = packed_buffer.data() +
                                     static_cast<size_t>(sp) * static_cast<size_t>(packed_len);
            uint8_t* dst_row = filtered_buffer.data() +
                               static_cast<size_t>(sp) * static_cast<size_t>(filtered_packed_len);
            for (int dst_pos = 0; dst_pos < effective_sequence_length; ++dst_pos) {
                const uint8_t state = unpack_state(src_row, kept_positions[static_cast<size_t>(dst_pos)]);
                const int packed_idx = dst_pos >> 1;
                if (dst_pos & 1) {
                    dst_row[packed_idx] =
                        static_cast<uint8_t>((dst_row[packed_idx] & 0x0F) | (state << 4));
                } else {
                    dst_row[packed_idx] =
                        static_cast<uint8_t>((dst_row[packed_idx] & 0xF0) | state);
                }
            }
        }
        packed_buffer.swap(filtered_buffer);
    }

    const int effective_packed_len = (effective_sequence_length + 1) / 2;
    auto* raw_buffer = new std::vector<uint8_t>(std::move(packed_buffer));
    py::capsule free_when_done(raw_buffer, [](void* v) {
        delete reinterpret_cast<std::vector<uint8_t>*>(v);
    });

    auto tensor = py::array_t<uint8_t>(
        {n_species, effective_packed_len},
        {
            static_cast<py::ssize_t>(effective_packed_len * static_cast<int>(sizeof(uint8_t))),
            static_cast<py::ssize_t>(sizeof(uint8_t)),
        },
        raw_buffer->data(),
        free_when_done
    );

    return py::make_tuple(tensor, species_names, effective_sequence_length);
}

py::array_t<int8_t> unpack_packed_tensor(py::array_t<uint8_t> packed, int sequence_length) {
    auto packed_buf = packed.request();
    if (packed_buf.ndim != 2) {
        throw std::runtime_error("packed tensor must have shape (n_species, packed_length)");
    }

    const int n_species = static_cast<int>(packed_buf.shape[0]);
    const int packed_len = static_cast<int>(packed_buf.shape[1]);
    const auto* packed_ptr = static_cast<const uint8_t*>(packed_buf.ptr);

    auto result = py::array_t<int8_t>({n_species, sequence_length});
    auto out = result.mutable_unchecked<2>();
    for (int sp = 0; sp < n_species; ++sp) {
        const uint8_t* row = packed_ptr + static_cast<size_t>(sp) * static_cast<size_t>(packed_len);
        for (int pos = 0; pos < sequence_length; ++pos) {
            out(sp, pos) = static_cast<int8_t>(unpack_state(row, pos));
        }
    }
    return result;
}

py::array_t<uint8_t> extract_packed_positions(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> packed,
    int sequence_length,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> positions,
    int workers
) {
    if (packed.ndim() != 2 || positions.ndim() != 2 ||
        sequence_length <= 0 || workers <= 0 ||
        packed.shape(1) < (sequence_length + 1) / 2) {
        throw std::runtime_error("invalid packed matrix, positions, length, or workers");
    }
    const py::ssize_t views = positions.shape(0);
    const py::ssize_t species = packed.shape(0);
    const py::ssize_t sites = positions.shape(1);
    const int64_t* selected = positions.data();
    for (py::ssize_t view = 0; view < views; ++view) {
        int64_t previous = -1;
        for (py::ssize_t site = 0; site < sites; ++site) {
            const int64_t position = selected[view * sites + site];
            if (position < 0 || position >= sequence_length || position <= previous) {
                throw std::runtime_error("positions must be strictly increasing and in range");
            }
            previous = position;
        }
    }
    py::array_t<uint8_t> output({views, species, sites});
    const uint8_t* source = packed.data();
    uint8_t* target = output.mutable_data();
    const py::ssize_t packed_width = packed.shape(1);
    {
        py::gil_scoped_release release;
        auto extract_row = [&](std::size_t index) {
            const py::ssize_t view = static_cast<py::ssize_t>(index) / species;
            const py::ssize_t taxon = static_cast<py::ssize_t>(index) % species;
            const uint8_t* row = source + taxon * packed_width;
            uint8_t* destination = target + (view * species + taxon) * sites;
            for (py::ssize_t site = 0; site < sites; ++site) {
                const uint8_t state = unpack_state(
                    row,
                    static_cast<int>(selected[view * sites + site])
                );
                destination[site] = state < 4 ? state : uint8_t{255};
            }
        };
        joined_parallel_for(
            static_cast<std::size_t>(views * species), workers, extract_row
        );
    }
    return output;
}

py::tuple filter_conserved_packed(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> packed,
    int sequence_length,
    int workers
) {
    if (packed.ndim() != 2 || sequence_length <= 0 || workers <= 0 ||
        packed.shape(1) < (sequence_length + 1) / 2) {
        throw std::runtime_error("invalid packed matrix, length, or workers");
    }
    const py::ssize_t species = packed.shape(0);
    const py::ssize_t packed_width = packed.shape(1);
    const uint8_t* source = packed.data();
    std::vector<uint8_t> keep(static_cast<std::size_t>(sequence_length), 0);
    {
        py::gil_scoped_release release;
        auto classify = [&](std::size_t raw_position) {
            const int position = static_cast<int>(raw_position);
            const uint8_t first = unpack_state(source, position);
            bool same = true;
            for (py::ssize_t taxon = 1; taxon < species; ++taxon) {
                if (unpack_state(source + taxon * packed_width, position) != first) {
                    same = false;
                    break;
                }
            }
            keep[static_cast<std::size_t>(position)] = same ? uint8_t{0} : uint8_t{1};
        };
        joined_parallel_for(
            static_cast<std::size_t>(sequence_length), workers, classify
        );
    }
    std::vector<int> kept_positions;
    kept_positions.reserve(static_cast<std::size_t>(sequence_length));
    for (int position = 0; position < sequence_length; ++position) {
        if (keep[static_cast<std::size_t>(position)] != 0) {
            kept_positions.push_back(position);
        }
    }
    const int effective_length = static_cast<int>(kept_positions.size());
    const py::ssize_t filtered_width = (effective_length + 1) / 2;
    auto* buffer = new std::vector<uint8_t>(
        static_cast<std::size_t>(species * filtered_width),
        kPackedGapByte
    );
    {
        py::gil_scoped_release release;
        auto pack_row = [&](std::size_t raw_taxon) {
            const py::ssize_t taxon = static_cast<py::ssize_t>(raw_taxon);
            const uint8_t* row = source + taxon * packed_width;
            uint8_t* destination = buffer->data() + taxon * filtered_width;
            for (int output_position = 0; output_position < effective_length; ++output_position) {
                const uint8_t state = unpack_state(
                    row,
                    kept_positions[static_cast<std::size_t>(output_position)]
                );
                const py::ssize_t byte = output_position >> 1;
                if (output_position & 1) {
                    destination[byte] = static_cast<uint8_t>(
                        (destination[byte] & 0x0F) | (state << 4)
                    );
                } else {
                    destination[byte] = static_cast<uint8_t>(
                        (destination[byte] & 0xF0) | state
                    );
                }
            }
        };
        joined_parallel_for(static_cast<std::size_t>(species), workers, pack_row);
    }
    py::capsule free_when_done(buffer, [](void* value) {
        delete reinterpret_cast<std::vector<uint8_t>*>(value);
    });
    auto output = py::array_t<uint8_t>(
        {species, filtered_width},
        {filtered_width, py::ssize_t{1}},
        buffer->data(),
        free_when_done
    );
    return py::make_tuple(output, effective_length);
}

py::tuple filter_interleaved_conserved_packed(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> packed,
    int sequence_length,
    int block_count,
    int workers
) {
    if (packed.ndim() != 2 || sequence_length <= 0 || block_count <= 0 ||
        block_count > sequence_length || workers <= 0 ||
        packed.shape(1) < (sequence_length + 1) / 2) {
        throw std::runtime_error(
            "invalid packed matrix, length, block count, or workers"
        );
    }
    const py::ssize_t species = packed.shape(0);
    const py::ssize_t packed_width = packed.shape(1);
    const uint8_t* source = packed.data();
    std::vector<uint8_t> keep(static_cast<std::size_t>(sequence_length), 0);
    {
        py::gil_scoped_release release;
        auto classify = [&](std::size_t raw_position) {
            const int position = static_cast<int>(raw_position);
            const uint8_t first = unpack_state(source, position);
            bool same = true;
            for (py::ssize_t taxon = 1; taxon < species; ++taxon) {
                if (unpack_state(source + taxon * packed_width, position) != first) {
                    same = false;
                    break;
                }
            }
            keep[raw_position] = same ? uint8_t{0} : uint8_t{1};
        };
        joined_parallel_for(
            static_cast<std::size_t>(sequence_length), workers, classify
        );
    }
    std::vector<std::vector<int>> kept_positions(
        static_cast<std::size_t>(block_count)
    );
    for (int position = 0; position < sequence_length; ++position) {
        if (keep[static_cast<std::size_t>(position)] != 0) {
            kept_positions[static_cast<std::size_t>(position % block_count)]
                .push_back(position);
        }
    }

    py::list outputs;
    py::list lengths;
    for (int block = 0; block < block_count; ++block) {
        const auto& selected = kept_positions[static_cast<std::size_t>(block)];
        const int effective_length = static_cast<int>(selected.size());
        const py::ssize_t filtered_width = (effective_length + 1) / 2;
        auto* buffer = new std::vector<uint8_t>(
            static_cast<std::size_t>(species * filtered_width),
            kPackedGapByte
        );
        {
            py::gil_scoped_release release;
            auto pack_row = [&](std::size_t raw_taxon) {
                const py::ssize_t taxon = static_cast<py::ssize_t>(raw_taxon);
                const uint8_t* row = source + taxon * packed_width;
                uint8_t* destination = buffer->data() + taxon * filtered_width;
                for (int output_position = 0;
                     output_position < effective_length;
                     ++output_position) {
                    const uint8_t state = unpack_state(
                        row,
                        selected[static_cast<std::size_t>(output_position)]
                    );
                    const py::ssize_t byte = output_position >> 1;
                    if (output_position & 1) {
                        destination[byte] = static_cast<uint8_t>(
                            (destination[byte] & 0x0F) | (state << 4)
                        );
                    } else {
                        destination[byte] = static_cast<uint8_t>(
                            (destination[byte] & 0xF0) | state
                        );
                    }
                }
            };
            joined_parallel_for(
                static_cast<std::size_t>(species), workers, pack_row
            );
        }
        py::capsule free_when_done(buffer, [](void* value) {
            delete reinterpret_cast<std::vector<uint8_t>*>(value);
        });
        outputs.append(py::array_t<uint8_t>(
            {species, filtered_width},
            {filtered_width, py::ssize_t{1}},
            buffer->data(),
            free_when_done
        ));
        lengths.append(effective_length);
    }
    return py::make_tuple(outputs, lengths);
}

py::tuple load_phy_to_tensor(const std::string& phy_path, bool drop_conserved_sites = false) {
    auto packed_out = load_phy_to_packed_tensor(phy_path, drop_conserved_sites);
    auto packed = packed_out[0].cast<py::array_t<uint8_t>>();
    auto species_names = packed_out[1].cast<std::vector<std::string>>();
    const int sequence_length = packed_out[2].cast<int>();
    auto dense = unpack_packed_tensor(packed, sequence_length);
    return py::make_tuple(dense, species_names);
}

py::object compute_quartet_indices(const py::object& pkl_or_obj, bool as_torch = true) {
    py::object sample_quartets;

    if (py::isinstance<py::str>(pkl_or_obj)) {
        auto pickle = py::module_::import("pickle");
        auto builtins = py::module_::import("builtins");
        py::object f = builtins.attr("open")(pkl_or_obj, "rb");
        try {
            sample_quartets = pickle.attr("load")(f);
        } catch (...) {
            try {
                f.attr("close")();
            } catch (...) {
            }
            throw;
        }
        f.attr("close")();
    } else if (py::isinstance<py::bytes>(pkl_or_obj)) {
        auto pickle = py::module_::import("pickle");
        sample_quartets = pickle.attr("loads")(pkl_or_obj);
    } else {
        sample_quartets = pkl_or_obj;
    }

    py::sequence seq = sample_quartets.cast<py::sequence>();
    const py::ssize_t n = seq.size();

    py::array_t<long long> out({n, static_cast<py::ssize_t>(4)});
    auto out_buf = out.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < n; ++i) {
        py::sequence quartet = seq[i].cast<py::sequence>();
        if (quartet.size() != 4) {
            throw std::runtime_error("Each quartet must have 4 species strings");
        }
        for (py::ssize_t j = 0; j < 4; ++j) {
            std::string name = py::str(quartet[j]);
            out_buf(i, j) = parse_species_index(name);
        }
    }

    if (!as_torch) {
        return out;
    }

    py::object torch = py::module_::import("torch");
    return torch.attr("from_numpy")(out).attr("to")(torch.attr("int64")).attr("clone")();
}

#ifndef SEQUENCE_PROCESSOR_BACKEND_MODULE_NAME
#define SEQUENCE_PROCESSOR_BACKEND_MODULE_NAME sequence_processor_backend
#endif

PYBIND11_MODULE(SEQUENCE_PROCESSOR_BACKEND_MODULE_NAME, m) {
    m.def(
        "load_phy_to_packed_tensor",
        &load_phy_to_packed_tensor,
        py::arg("phy_path"),
        py::arg("drop_conserved_sites") = false,
        "Load PHY file and return (packed_sequences_tensor, species_names, sequence_length)"
    );
    m.def(
        "unpack_packed_tensor",
        &unpack_packed_tensor,
        py::arg("packed"),
        py::arg("sequence_length"),
        "Unpack uint8 nibble-packed sequence tensor to int8 states"
    );
    m.def(
        "load_phy_to_tensor",
        &load_phy_to_tensor,
        py::arg("phy_path"),
        py::arg("drop_conserved_sites") = false,
        "Load PHY file and return dense tensor (n_species, seq_length) and species names"
    );
    m.def(
        "extract_packed_positions",
        &extract_packed_positions,
        py::arg("packed"),
        py::arg("sequence_length"),
        py::arg("positions"),
        py::arg("workers") = 1,
        "Extract several ordered site sketches from one packed alignment"
    );
    m.def(
        "filter_conserved_packed",
        &filter_conserved_packed,
        py::arg("packed"),
        py::arg("sequence_length"),
        py::arg("workers") = 1,
        "Drop raw-constant sites from an existing packed alignment"
    );
    m.def(
        "filter_interleaved_conserved_packed",
        &filter_interleaved_conserved_packed,
        py::arg("packed"),
        py::arg("sequence_length"),
        py::arg("block_count"),
        py::arg("workers") = 1,
        "Partition by original site modulo block_count and drop constants"
    );
    m.def(
        "compute_quartet_indices",
        &compute_quartet_indices,
        py::arg("pkl_or_obj"),
        py::arg("as_torch") = true,
        "Compute quartet indices from a pickle path/bytes/object"
    );
}
