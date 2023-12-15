#include "selfattention.hh"
#include "memory.h"
#include <cmath>
#include <iostream>

template<typename T>
SingleHeadSelfAttn<T>::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       Matrix<T> **weightVector) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;

    query_layer = new Dense<T>(input_dim, head_hidden_size, weightVector[0]);
    key_layer = new Dense<T>(input_dim, head_hidden_size, weightVector[1]);
    value_layer = new Dense<T>(input_dim, head_hidden_size, weightVector[2]);
    softmax = new Softmax<T>();

    query_layer_out = new Matrix<T>(pre_seq_len, head_hidden_size);
    key_layer_out = new Matrix<T>(pre_seq_len ,head_hidden_size);
    value_layer_out = new Matrix<T>(pre_seq_len ,head_hidden_size);
    value_transposed_layer_out = new Matrix<T>(head_hidden_size, pre_seq_len);
    attention_scores = new Matrix<T>(pre_seq_len ,pre_seq_len);
}

template<typename T>
SingleHeadSelfAttn<T>::~SingleHeadSelfAttn() {

    delete query_layer_out;
    delete key_layer_out;
    delete value_transposed_layer_out;
    delete value_layer_out;
    delete attention_scores;

    delete query_layer;
    delete key_layer;
    delete value_layer;
    delete softmax;
}

template<typename T>
void SingleHeadSelfAttn<T>::compute(std::size_t seq_len, Matrix<T> &input, Matrix<T> &output) {

    // Input shape [seq_len, input_dim] times [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    // From [seq_len, input_dim] to [seq_len, head_hidden_size]
    // Output shape [seq_len, head_hidden_size]
    // seq_len = 512; input_dim = 1024; head_hidden_size = 64; the number of heads = 16
    query_layer->compute(seq_len, input, *query_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    // print_tensor(*query_layer_out);
    key_layer->compute(seq_len, input, *key_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    value_layer->compute(seq_len, input, *value_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]

    // // Not sure
    // Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_, pre_seq_len_);//[seq_len, head_hidden_size] -> [head_hidden_size, seq_len]

    Matrix<T>::MatrixCompute(seq_len, *query_layer_out, *attention_scores, *key_layer_out, head_hidden_size_,
                            seq_len);//[seq_len, head_hidden_size] * [head_hidden_size, seq_len] = [seq_len, seq_len]
    // print_tensor(*attention_scores);
    softmax->compute(*attention_scores, seq_len);//[seq_len, seq_len]
    // print_tensor(*attention_scores);
    Matrix<T>::transpose(*value_layer_out, *value_transposed_layer_out);//[seq_len, head_hidden_size] -> [head_hidden_size, seq_len]
    // print_tensor(*value_transposed_layer_out);
    // print_tensor(*attention_scores);
    Matrix<T>::MatrixCompute(seq_len, *attention_scores, output, *value_transposed_layer_out, seq_len, head_hidden_size_);//[seq_len, seq_len] * [seq_len, head_hidden_size] = [seq_len, head_hidden_size]
    // softmax->post_softmax(output, seq_len);//[seq_len, head_hidden_size]
    // print_tensor(output);
}

template class SingleHeadSelfAttn<int>;
template class SingleHeadSelfAttn<float>;
template class SingleHeadSelfAttn<u_int32_t>;