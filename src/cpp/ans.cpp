#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <algorithm>

namespace py = pybind11;

const uint64_t RANS_L = 1ULL << 31;

struct Instruction {
    uint64_t start;
    uint64_t freq;
    bool flag;
};

using message_t = std::pair<py::array_t<uint64_t>, py::list>;

message_t push(message_t x, uint64_t start, uint64_t freq, int precision) {
    uint64_t x_head = *x.first.data();
    
    uint64_t x_max = ((RANS_L >> precision) << 32) * freq;
    if (x_head >= x_max) {
        x.second.append(uint32_t(x_head & 0xFFFFFFFF));
        x_head >>= 32;
    }
    
    *x.first.mutable_data() = ((x_head / freq) << precision) + (x_head % freq) + start;
    return x;
}

std::pair<message_t, uint64_t> pop(message_t x, int precision) {
    uint64_t head_val = *x.first.data();
    uint64_t interval_starts = head_val & ((1ULL << precision) - 1);

    // Renormalization
    if (head_val < RANS_L) {
        py::list tail_list = x.second;
        if (tail_list.size() > 0) {
            uint32_t new_head_part = tail_list[0].cast<uint32_t>();
            py::list new_tail_list;
            for (size_t i = 1; i < tail_list.size(); ++i) {
                new_tail_list.append(tail_list[i]);
            }
            x.second = new_tail_list;
            head_val = (head_val << 32) | new_head_part;
            *x.first.mutable_data() = head_val;
        }
    }
    return std::make_pair(x, interval_starts);
}

std::pair<std::vector<Instruction>, py::tuple> ans_index_buffered_encoder(
    py::array_t<int32_t> symbols,
    py::array_t<int32_t> indices,
    py::array_t<uint64_t> cdf,
    py::array_t<int32_t> cdf_length,
    py::array_t<int32_t> cdf_offset,
    int precision,
    int overflow_width) {

    py::buffer_info symbols_buf = symbols.request();
    py::buffer_info indices_buf = indices.request();
    py::buffer_info cdf_buf = cdf.request();
    py::buffer_info cdf_length_buf = cdf_length.request();
    py::buffer_info cdf_offset_buf = cdf_offset.request();

    int32_t *symbols_ptr = static_cast<int32_t *>(symbols_buf.ptr);
    int32_t *indices_ptr = static_cast<int32_t *>(indices_buf.ptr);
    uint64_t *cdf_ptr = static_cast<uint64_t *>(cdf_buf.ptr);
    int32_t *cdf_length_ptr = static_cast<int32_t *>(cdf_length_buf.ptr);
    int32_t *cdf_offset_ptr = static_cast<int32_t *>(cdf_offset_buf.ptr);

    ssize_t n_symbols = symbols_buf.shape[0];
    ssize_t n_cdfs = cdf_buf.shape[0];
    ssize_t cdf_max_len = cdf_buf.shape[1];

    std::vector<Instruction> instructions;

    uint64_t max_overflow = (1ULL << overflow_width) - 1;

    for (ssize_t i = 0; i < n_symbols; ++i) {
        int32_t cdf_index = indices_ptr[i];
        int32_t cdf_len = cdf_length_ptr[cdf_index];
        int32_t max_value = cdf_len - 2;

        int32_t value = symbols_ptr[i];
        value -= cdf_offset_ptr[cdf_index];

        uint64_t overflow = 0;
        if (value < 0) {
            overflow = -2 * value - 1;
            value = max_value;
        } else if (value >= max_value) {
            overflow = 2 * (value - max_value);
            value = max_value;
        }

        uint64_t *cdf_i_ptr = cdf_ptr + cdf_index * cdf_max_len;
        uint64_t start = cdf_i_ptr[value];
        uint64_t freq = cdf_i_ptr[value + 1] - start;
        instructions.push_back({start, freq, false});

        if (value == max_value) {
            int widths = 0;
            if (overflow > 0) {
                widths = (64 - __builtin_clzll(overflow) + overflow_width - 1) / overflow_width;
            }

            uint64_t val = widths;
            while (val >= max_overflow) {
                instructions.push_back({max_overflow, 1, true});
                val -= max_overflow;
            }
            instructions.push_back({val, 1, true});

            for (int j = 0; j < widths; ++j) {
                val = (overflow >> (j * overflow_width)) & max_overflow;
                instructions.push_back({val, 1, true});
            }
        }
    }
    
    py::tuple coding_shape = py::make_tuple(symbols_buf.shape[1], symbols_buf.shape[2], symbols_buf.shape[3]); // Assuming 4D input N,C,H,W

    return std::make_pair(instructions, coding_shape);
}

py::array_t<uint32_t> ans_index_encoder_flush(
    std::vector<Instruction> instructions,
    int precision,
    int overflow_width) {
    
    auto head = py::array_t<uint64_t>(1);
    *head.mutable_data() = RANS_L;
    auto tail = py::list();
    message_t message = std::make_pair(head, tail);

    std::reverse(instructions.begin(), instructions.end());

    for (const auto& instruction : instructions) {
        if (!instruction.flag) {
            message = push(message, instruction.start, instruction.freq, precision);
        } else {
            // This is not quite right, need to check the python implementation
            message = push(message, instruction.start, instruction.freq, overflow_width);
        }
    }

    py::list flat_message;
    uint64_t head_val = *message.first.data();
    flat_message.append(uint32_t(head_val >> 32));
    flat_message.append(uint32_t(head_val & 0xFFFFFFFF));
    
    for (ssize_t i = message.second.size() - 1; i >= 0; --i) {
        flat_message.append(message.second[i]);
    }
    
    py::array_t<uint32_t> result(flat_message.size());
    py::buffer_info buf = result.request();
    uint32_t *ptr = static_cast<uint32_t *>(buf.ptr);
    for (size_t i = 0; i < flat_message.size(); i++) {
        ptr[i] = flat_message[i].cast<uint32_t>();
    }

    return result;
}

message_t unflatten_scalar(py::array_t<uint32_t> arr) {
    py::buffer_info arr_buf = arr.request();
    uint32_t *arr_ptr = static_cast<uint32_t *>(arr_buf.ptr);

    auto head = py::array_t<uint64_t>(1);
    *head.mutable_data() = (uint64_t(arr_ptr[0]) << 32) | uint64_t(arr_ptr[1]);

    py::list tail_list;
    for (ssize_t i = arr_buf.shape[0] - 1; i >= 2; --i) {
        tail_list.append(arr_ptr[i]);
    }
    return std::make_pair(head, tail_list);
}


py::array_t<int32_t> ans_index_decoder(
    py::array_t<uint32_t> encoded,
    py::array_t<int32_t> indices,
    py::array_t<uint64_t> cdf,
    py::array_t<int32_t> cdf_length,
    py::array_t<int32_t> cdf_offset,
    int precision,
    int overflow_width) {

    py::buffer_info indices_buf = indices.request();
    py::buffer_info cdf_buf = cdf.request();
    py::buffer_info cdf_length_buf = cdf_length.request();
    py::buffer_info cdf_offset_buf = cdf_offset.request();

    int32_t *indices_ptr = static_cast<int32_t *>(indices_buf.ptr);
    uint64_t *cdf_ptr = static_cast<uint64_t *>(cdf_buf.ptr);
    int32_t *cdf_length_ptr = static_cast<int32_t *>(cdf_length_buf.ptr);
    int32_t *cdf_offset_ptr = static_cast<int32_t *>(cdf_offset_buf.ptr);

    ssize_t n_symbols = indices_buf.shape[0];
    ssize_t cdf_max_len = cdf_buf.shape[1];

    py::array_t<int32_t> decoded_symbols(n_symbols);
    int32_t *decoded_symbols_ptr = static_cast<int32_t *>(decoded_symbols.request().ptr);

    message_t message = unflatten_scalar(encoded);

    uint64_t max_overflow = (1ULL << overflow_width) - 1;

    for (ssize_t i = n_symbols - 1; i >= 0; --i) {
        int32_t cdf_index = indices_ptr[i];
        int32_t cdf_len = cdf_length_ptr[cdf_index];
        int32_t max_value_in_range = cdf_len - 2;

        uint64_t *cdf_i_ptr = cdf_ptr + cdf_index * cdf_max_len;
        
        auto popped = pop(message, precision);
        message = popped.first;
        uint64_t cf = popped.second; // This is the cum_freq in python code

        // Replicate np.searchsorted(cdf_i, cf, side='right') - 1
        auto it = std::upper_bound(cdf_i_ptr, cdf_i_ptr + cdf_len, cf);
        int32_t value = std::distance(cdf_i_ptr, it) - 1;
        
        // This is a correction based on the python code.
        uint64_t start = cdf_i_ptr[value];
        uint64_t freq = cdf_i_ptr[value + 1] - start;

        // update head for next pop
        *message.first.mutable_data() = (cf / freq) * (1ULL << precision) + (cf % freq) - start;

        if (value == max_value_in_range) {
            // Handle overflow
            popped = pop(message, overflow_width);
            message = popped.first;
            uint64_t val = popped.second; // First val is widths
            int widths = val;

            while (val == max_overflow) {
                popped = pop(message, overflow_width);
                message = popped.first;
                val = popped.second;
                widths += val;
            }

            uint64_t overflow_val = 0;
            for (int j = 0; j < widths; ++j) {
                popped = pop(message, overflow_width);
                message = popped.first;
                val = popped.second;
                overflow_val |= (val << (j * overflow_width));
            }

            // Decode the actual value from overflow
            value = overflow_val >> 1;
            if ((overflow_val & 1)) { // if odd
                value = -value - 1;
            } else { // if even
                value += max_value_in_range;
            }
        }
        decoded_symbols_ptr[i] = static_cast<int32_t>(value + cdf_offset_ptr[cdf_index]);
    }
    return decoded_symbols;
}

PYBIND11_MODULE(hific_cpp, m) {
    m.doc() = "C++ implementation of ANS coding";
    py::class_<Instruction>(m, "Instruction")
        .def(py::init<>())
        .def_readwrite("start", &Instruction::start)
        .def_readwrite("freq", &Instruction::freq)
        .def_readwrite("flag", &Instruction::flag);
    m.def("ans_index_buffered_encoder", &ans_index_buffered_encoder, "ANS buffered encoder");
    m.def("ans_index_encoder_flush", &ans_index_encoder_flush, "ANS encoder flush");
    m.def("ans_index_decoder", &ans_index_decoder, "ANS decoder");
}
