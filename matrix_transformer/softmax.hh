#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>

#include "matrix_data.hh"

template<typename T>
class Softmax
{
    public:
        explicit Softmax();
        ~Softmax();
        void compute(Matrix<T> &input, std::size_t seq_len);
        // void post_softmax(Matrix<T> &input, size_t seq_len);
    private:

};