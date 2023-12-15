#pragma once
#include "selfattention.hh"
#include "addNorm.hh"
#include <fstream>
#include <iostream>

template<typename T>
class TransformerBlock{
public:
    TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                     std::size_t ff_size, Matrix<T> **weightVector);

    virtual ~TransformerBlock();

    void compute(std::size_t seq_len, Matrix<T> &input, Matrix<T> &output);

public:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t ff_size_;
    SingleHeadSelfAttn<T>* selfatten[16];
    Matrix<T>** multihead_out;
    Matrix<T>* condense_out;
    Matrix<T>* intermediateFF;
    AddNormalize<T>* addNorm;
    Dense<T>* condense;
    Dense<T>* feedForward0;
    Dense<T>* feedForward1;

};