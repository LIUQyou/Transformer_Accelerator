//
// Created by alireza on 3/2/22.
//
#pragma once
#include "util.h"
#include "iostream"
#ifndef FVLLMONTITRANSFORMER_ADDNORM_H
#define FVLLMONTITRANSFORMER_ADDNORM_H

class AddNormalize{
public:
    AddNormalize(std::size_t, std::size_t);
    void compute(uint32_t *input, uint32_t *output);
private:
    std::size_t seq_len_;
    std::size_t input_dim_;

};

#endif //FVLLMONTITRANSFORMER_ADDNORM_H
