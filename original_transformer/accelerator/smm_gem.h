//
// Created by alireza on 3/3/22.
//

#ifndef FVLLMONTITRANSFORMER_SMM_GEM_H
#define FVLLMONTITRANSFORMER_SMM_GEM_H

void MatrixCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_);
void conventionalCompute(std::size_t seq_len, const uint32_t * input, uint32_t * output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_);
void SystolicCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weight,
                         std::size_t input_size_, std::size_t output_size_);


/// @brief Matrix multiplication using accelerator
/// @param seq_len 
/// @param input 
/// @param output 
/// @param weights 
/// @param input_size_ 
/// @param output_size_ 
void smmCompute(std::size_t seq_len, const uint32_t *input, uint32_t *output, uint32_t *weights,
                std::size_t input_size_, std::size_t output_size_);

void print_arr(uint32_t* array, int n, int p);

#endif //FVLLMONTITRANSFORMER_SMM_GEM_H