#include "dense.hh"
#include <exception>
//#include <mkl.h>
#include <memory.h>
#include <iostream>

template<typename T>
Dense<T>::Dense(std::size_t input_size, std::size_t output_size, Matrix<T> *weightDense) {
    input_size_  = input_size;
    output_size_ = output_size;
    std::cout << "Input Size : " << input_size_ << std::endl;
    std::cout << "Output Size : " << output_size_ << std::endl;
    weight = weightDense;
    bias = nullptr;
}

template<typename T>
Dense<T>::~Dense() {
//    delete weight;
//    delete[] bias;
}

template<typename T>
void Dense<T>::multiplyweight(std::size_t seq_len, Matrix<T>&input, Matrix<T> &output) {
    Matrix<T>::MatrixCompute(seq_len, input, output, *weight, input_size_, output_size_);
}

template<typename T>
void Dense<T>::addbias(std::size_t seq_len, Matrix<T> &output) {

    for (std::size_t idx = 0; idx < seq_len; idx++) {
        for (std::size_t feature_idx = 0; feature_idx < output_size_; feature_idx++) {
            // output[idx * output_size_ + feature_idx] += bias[feature_idx];
            output.at(idx, feature_idx) += bias[feature_idx];
        }
    }
}

template<typename T>
void Dense<T>::compute(std::size_t seq_len, Matrix<T> &input, Matrix<T> &output) {
    // input shape [batch_size, input_size_]
    // output shape [batch_size, output_size_]

    multiplyweight(seq_len, input, output);
    // add bias vector here
    if (bias != nullptr) {
        addbias(seq_len, output);
    }
}


template class Dense<int>;
template class Dense<float>;
template class Dense<u_int32_t>;