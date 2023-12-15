#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>

#include "matrix_data.hh"

template<typename T>
class AddNormalize{
public:
    AddNormalize(std::size_t, std::size_t);
    void compute(Matrix<T> &input, Matrix<T> &output);
private:
    std::size_t seq_len_;
    std::size_t input_dim_;

};