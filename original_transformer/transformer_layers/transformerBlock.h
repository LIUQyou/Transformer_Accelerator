//
// Created by alireza on 3/2/22.
//
#pragma once
#include <fstream>
#include <iostream>
#include "selfattention.h"
#include "addNorm.h"
#include "dense.h"
#include "util.h"


#ifndef FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
#define FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H

class TransformerBlock{
public:
    TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                     std::size_t ff_size, uint32_t ** weightVector);

    virtual ~TransformerBlock();

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output, uint32_t* &temp);

public:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t ff_size_;
    SingleHeadSelfAttn* selfatten[16];
    uint32_t* multihead_out;
    uint32_t* condense_out;
    uint32_t* intermediateFF;
    uint32_t* intermediateFF_debug;
    AddNormalize* addNorm;
    Dense* condense;
    Dense* feedForward0;
    Dense* feedForward1;

};

class TransformerBlockSystolic{
public:
    TransformerBlockSystolic(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                     std::size_t ff_size, uint32_t ** weightVector);

    virtual ~TransformerBlockSystolic();

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output, uint32_t* &temp);

public:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t ff_size_;
    SingleHeadSelfAttnSystolic* selfatten[16];
    uint32_t* multihead_out;
    uint32_t* condense_out;
    uint32_t* intermediateFF;
    uint32_t* intermediateFF_debug;
    AddNormalize* addNorm;
    Dense* condense;
    Dense* feedForward0;
    Dense* feedForward1;

};

#endif //FVLLMONTITRANSFORMER_MULTIHEADSELFATTENTION_H
