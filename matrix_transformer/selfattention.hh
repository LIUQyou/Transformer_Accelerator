#pragma once

#include "matrix_data.hh"
#include "dense.hh"
#include "softmax.hh"
#include "utils.hh"

template<typename T>
class SingleHeadSelfAttn{
    public:
        SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim_, std::size_t head_hidden_size, Matrix<T> **weightVector);
        ~SingleHeadSelfAttn();
        void compute(std::size_t seq_len, Matrix<T> &input, Matrix<T> &output);

    private:
        Dense<T>* query_layer;
        Dense<T>* key_layer;
        Dense<T>* value_layer;
        Softmax<T>* softmax;

        Matrix<T> *query_layer_out;
        Matrix<T> *key_layer_out;
        Matrix<T> *value_layer_out;
        Matrix<T> *value_transposed_layer_out;
        Matrix<T> *attention_scores;

        std::size_t pre_seq_len_;
        std::size_t head_hidden_size_;
};