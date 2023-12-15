#include "selfattention.h"
#include "memory.h"
#include "../accelerator/smm_gem.h"
#include <cmath>
#include <iostream>

SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t ** weightVector) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;

    query_layer = new Dense(input_dim, head_hidden_size, weightVector[0]);
    key_layer = new Dense(input_dim, head_hidden_size, weightVector[1]);
    value_layer = new Dense(input_dim, head_hidden_size, weightVector[2]);
    softmax = new Softmax();

    query_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    key_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    value_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    attention_scores = new uint32_t[pre_seq_len * pre_seq_len ];
}

SingleHeadSelfAttn::~SingleHeadSelfAttn() {

    delete[] query_layer_out;
    delete[] key_layer_out;
    delete[] key_transposed_layer_out;
    delete[] value_layer_out;
    delete[] attention_scores;

    delete query_layer;
    delete key_layer;
    delete value_layer;
    delete softmax;
}

void print_tensor(uint32_t *output, std::size_t seq_len, std::size_t head_hidden_size)
{
    for(std::size_t i = 0; i < head_hidden_size; i++) {
        for(std::size_t j = 0; j < seq_len; j++) {
            std::cout << output[i*head_hidden_size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void SingleHeadSelfAttn::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {

    // Input shape [seq_len, input_dim] times [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    // From [seq_len, input_dim] to [seq_len, head_hidden_size]
    // Output shape [seq_len, head_hidden_size]
    // seq_len = 512; input_dim = 1024; head_hidden_size = 64; the number of heads = 16
    query_layer->compute(seq_len, input, query_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    key_layer->compute(seq_len, input, key_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    value_layer->compute(seq_len, input, value_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_, pre_seq_len_);//[seq_len, head_hidden_size] -> [head_hidden_size, seq_len]
    MatrixCompute(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                            seq_len);//[seq_len, head_hidden_size] * [head_hidden_size, seq_len] = [seq_len, seq_len]
    softmax->compute(attention_scores, seq_len);//[seq_len, seq_len]
    MatrixCompute(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);//[seq_len, seq_len] * [seq_len, head_hidden_size] = [seq_len, head_hidden_size]
    // softmax->post_softmax(output, seq_len);//[seq_len, head_hidden_size]
    // print_tensor(output, seq_len, head_hidden_size_);
}

SingleHeadSelfAttnSystolic::SingleHeadSelfAttnSystolic(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                       uint32_t ** weightVector) {

    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;

    query_layer = new Dense(input_dim, head_hidden_size, weightVector[0]);
    key_layer = new Dense(input_dim, head_hidden_size, weightVector[1]);
    value_layer = new Dense(input_dim, head_hidden_size, weightVector[2]);
    softmax = new Softmax();

    query_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    key_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    key_transposed_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    value_layer_out = new uint32_t[pre_seq_len * head_hidden_size ];
    attention_scores = new uint32_t[pre_seq_len * pre_seq_len ];
}

SingleHeadSelfAttnSystolic::~SingleHeadSelfAttnSystolic() {

    delete[] query_layer_out;
    delete[] key_layer_out;
    delete[] key_transposed_layer_out;
    delete[] value_layer_out;
    delete[] attention_scores;

    delete query_layer;
    delete key_layer;
    delete value_layer;
    delete softmax;
}

void SingleHeadSelfAttnSystolic::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {

    // Input shape [seq_len, input_dim] times [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    // From [seq_len, input_dim] to [seq_len, head_hidden_size]
    // Output shape [seq_len, head_hidden_size]
    // seq_len = 512; input_dim = 1024; head_hidden_size = 64; the number of heads = 16
    query_layer->computeSystolic(seq_len, input, query_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    key_layer->computeSystolic(seq_len, input, key_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    value_layer->computeSystolic(seq_len, input, value_layer_out);//[seq_len, input_dim] * [input_dim, head_hidden_size] = [seq_len, head_hidden_size]
    Transpose::transpose(key_layer_out, key_transposed_layer_out, head_hidden_size_, pre_seq_len_);//[seq_len, head_hidden_size] -> [head_hidden_size, seq_len]
    SystolicCompute(seq_len, query_layer_out, attention_scores, key_transposed_layer_out, head_hidden_size_,
                            seq_len);//[seq_len, head_hidden_size] * [head_hidden_size, seq_len] = [seq_len, seq_len]
    softmax->compute(attention_scores, seq_len);//[seq_len, seq_len]
    SystolicCompute(seq_len, attention_scores, output, value_layer_out, seq_len, head_hidden_size_);//[seq_len, seq_len] * [seq_len, head_hidden_size] = [seq_len, head_hidden_size]
    // softmax->post_softmax(output, seq_len);//[seq_len, head_hidden_size]
}