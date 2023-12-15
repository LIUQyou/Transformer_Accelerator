#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>

#include "matrix_data.hh"

template<typename T>
class Dense {
public:
    Dense(std::size_t input_dim, std::size_t output_dim, Matrix<T> *weight);

    ~Dense();

    void compute(std::size_t seq_len, Matrix<T> &input, Matrix<T> &output);

private:
    void multiplyweight(std::size_t seq_len, Matrix<T> &input, Matrix<T> &output);

    void addbias(std::size_t seq_len, Matrix<T> &output);

    std::size_t input_size_;
    std::size_t output_size_;
    Matrix<T> *weight; // shape [input_size_, output_size_]
    uint32_t *bias;   // shape [output_size_]

};