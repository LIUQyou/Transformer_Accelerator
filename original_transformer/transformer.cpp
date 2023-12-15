
//#include"gtest/gtest.h"
#include "transformer.h"


void fill_kernel(uint32_t* kernel, int kernel_size){
    for(int i=0; i<kernel_size; i++){
        // uint32_t result = 0;
        // for (int j=0; j<4; j++){
        //     result |=  ((uint8_t)(rand() % 5  - 2)) << (8 * j);
        // }
        kernel[i]=i%5;
    }
}

void test(){
    std::cout<<"First line" << std::endl;
    uint32_t tensor_in[D_SEQ * D_MODEL];
    if(!readFromCSV("tensor_in.csv", tensor_in, D_MODEL, D_SEQ))
    {
        std::cout<<"Error: reading input file"<<std::endl;
        return;
    }
    uint32_t out[D_SEQ*D_MODEL];
    uint32_t out_systolic[D_SEQ*D_MODEL];
    uint32_t *temp;
    uint32_t *temp_systolic;


    uint32_t * weightVec[3*NUM_HEAD+3];

    for (int n=0; n<NUM_HEAD; n++){
        auto query_kernel = new uint32_t [D_Q* D_MODEL];
        fill_kernel(query_kernel, D_Q* D_MODEL);
        weightVec[n*3] = query_kernel;

        auto key_kernel = new uint32_t [ D_Q* D_MODEL];
        fill_kernel(key_kernel, D_Q* D_MODEL);
        weightVec[n*3 + 1] = key_kernel;

        auto value_kernel = new uint32_t [ D_Q* D_MODEL];
        fill_kernel(value_kernel, D_Q* D_MODEL);
        weightVec[n*3 + 2] = value_kernel;
    }

    uint32_t condense_kernel[ NUM_HEAD * D_Q * D_MODEL];
    fill_kernel(condense_kernel, NUM_HEAD * D_Q * D_MODEL);
    weightVec[NUM_HEAD*3] = condense_kernel;

    auto ff0_kernel = new uint32_t [ D_MODEL* D_FF];
    fill_kernel(ff0_kernel, D_MODEL* D_FF);
    weightVec[NUM_HEAD*3+1] = ff0_kernel;

    auto ff1_kernel = new uint32_t [ D_FF* D_MODEL];
    fill_kernel(ff1_kernel, D_FF* D_MODEL);
    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec);
    selfatten.compute(D_SEQ, tensor_in, out, temp);

    TransformerBlockSystolic selfattenSystolic(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec);
    selfattenSystolic.compute(D_SEQ, tensor_in, out_systolic, temp_systolic);

    //compare the results from systolic and tiled
    for(int i=0; i<D_SEQ*D_MODEL; i++){
        if(temp[i] != temp_systolic[i]){
            std::cout<<"Error: systolic and tiled results are different"<<std::endl;
            return;
        }
    }
    std::cout<<"Test finished!!" << std::endl;
}

void test_blocks(){
    uint32_t *tensor_in = new uint32_t[D_SEQ * D_MODEL];
    fill_kernel(tensor_in, D_SEQ * D_MODEL);
    uint32_t *out = new uint32_t[D_SEQ*D_MODEL];
    uint32_t *out_systolic = new uint32_t[D_SEQ*D_MODEL];
    uint32_t *temp;
    uint32_t *temp_systolic;


    uint32_t * weightVec[3*NUM_HEAD+3];

    for (int n=0; n<NUM_HEAD; n++){
        auto query_kernel = new uint32_t [D_Q* D_MODEL];
        fill_kernel(query_kernel, D_Q* D_MODEL);
        weightVec[n*3] = query_kernel;

        auto key_kernel = new uint32_t [ D_Q* D_MODEL];
        fill_kernel(key_kernel, D_Q* D_MODEL);
        weightVec[n*3 + 1] = key_kernel;

        auto value_kernel = new uint32_t [ D_Q* D_MODEL];
        fill_kernel(value_kernel, D_Q* D_MODEL);
        weightVec[n*3 + 2] = value_kernel;
    }

    uint32_t *condense_kernel = new uint32_t[ NUM_HEAD * D_Q * D_MODEL];
    fill_kernel(condense_kernel, NUM_HEAD * D_Q * D_MODEL);
    weightVec[NUM_HEAD*3] = condense_kernel;

    auto ff0_kernel = new uint32_t [ D_MODEL* D_FF];
    fill_kernel(ff0_kernel, D_MODEL* D_FF);
    weightVec[NUM_HEAD*3+1] = ff0_kernel;

    auto ff1_kernel = new uint32_t [ D_FF* D_MODEL];
    fill_kernel(ff1_kernel, D_FF* D_MODEL);
    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;

    uint32_t *intermediateFF = new uint32_t[D_SEQ * D_FF];
    uint32_t *intermediateFF_debug = new uint32_t[D_SEQ * D_FF];

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec);
    selfatten.feedForward0->compute(D_SEQ, tensor_in, intermediateFF);
    selfatten.feedForward0->computeSystolic(D_SEQ, tensor_in, intermediateFF_debug);
    // compare the results from systolic and tiled
    for(int i=0; i<D_SEQ*D_FF; i++){
        if(intermediateFF[i] != intermediateFF_debug[i]){
            std::cout<<"Error: systolic and tiled results are different"<<std::endl;
            return;
        }
    }
    selfatten.compute(D_SEQ, tensor_in, out, temp);

    TransformerBlockSystolic selfattenSystolic(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec);
    selfattenSystolic.feedForward0->compute(D_SEQ, tensor_in, intermediateFF);
    selfattenSystolic.feedForward0->computeSystolic(D_SEQ, tensor_in, intermediateFF_debug);
    // compare the results from systolic and tiled
    for(int i=0; i<D_SEQ*D_FF; i++){
        if(intermediateFF[i] != intermediateFF_debug[i]){
            std::cout<<"Error: systolic and tiled results are different"<<std::endl;
            return;
        }
    }
    selfattenSystolic.compute(D_SEQ, tensor_in, out_systolic, temp_systolic);

    //compare the results from systolic and tiled
    for(int i=0; i<D_SEQ*D_MODEL; i++){
        if(temp[i] != temp_systolic[i]){
            std::cout<<"Error: systolic and tiled results are different"<<std::endl;
            return;
        }
    }
    std::cout<<"Test finished!!" << std::endl;
}

void rand_init(uint32_t *input, int size){
    for (int i=0; i<size; i++){
        input[i] = rand() % 5;
    }
}

void test_systolic()
{
    int seq_len = 256;
    int input_size_ = 512;
    int output_size_ = 4096;
    uint32_t *input = new uint32_t[seq_len*input_size_];
    uint32_t *weight = new uint32_t[input_size_*output_size_];
    uint32_t *output_systolic = new uint32_t[seq_len*output_size_];
    uint32_t *output_tile = new uint32_t[seq_len*output_size_];
    rand_init(input, seq_len*input_size_);
    rand_init(weight, input_size_*output_size_);
    MatrixCompute(seq_len, input, output_systolic, weight, input_size_, output_size_);
    SystolicCompute(seq_len, input, output_tile, weight, input_size_, output_size_);
    for(int j=0; j<seq_len*output_size_; j++){
        if(output_systolic[j] != output_tile[j]){
            for(int i=0; i<input_size_; i++){
                std::cout<<weight[i* output_size_ + j%output_size_]<<",";
            }
            std::cout<<std::endl;
            for(int i=0; i<input_size_; i++){
                std::cout<<input[i + j/output_size_*input_size_]<<",";
            }
            std::cout<<"Error: systolic and tiled results are different"<<std::endl;
            return;
        }
    }

    delete[] input;
    delete[] weight;
    delete[] output_systolic;
    delete[] output_tile;
}


void run_transformer(){
    std::cout<<"First line" << std::endl;
    uint32_t tensor_in[D_SEQ * D_MODEL];
    if(!readFromCSV("tensor_in.csv", tensor_in, D_MODEL, D_SEQ))
    {
        std::cout<<"Error: reading input file"<<std::endl;
        return;
    }
    uint32_t out[D_SEQ*D_MODEL];
    uint32_t out_systolic[D_SEQ*D_MODEL];
    uint32_t *temp;
    uint32_t *temp_systolic;


    uint32_t * weightVec[3*NUM_HEAD+3];

    for (int n=0; n<NUM_HEAD; n++){
        auto query_kernel = new uint32_t [D_Q* D_MODEL];
        fill_kernel(query_kernel, D_Q* D_MODEL);
        weightVec[n*3] = query_kernel;

        auto key_kernel = new uint32_t [ D_Q* D_MODEL];
        fill_kernel(key_kernel, D_Q* D_MODEL);
        weightVec[n*3 + 1] = key_kernel;

        auto value_kernel = new uint32_t [ D_Q* D_MODEL];
        fill_kernel(value_kernel, D_Q* D_MODEL);
        weightVec[n*3 + 2] = value_kernel;
    }

    uint32_t condense_kernel[ NUM_HEAD * D_Q * D_MODEL];
    fill_kernel(condense_kernel, NUM_HEAD * D_Q * D_MODEL);
    weightVec[NUM_HEAD*3] = condense_kernel;

    auto ff0_kernel = new uint32_t [ D_MODEL* D_FF];
    fill_kernel(ff0_kernel, D_MODEL* D_FF);
    weightVec[NUM_HEAD*3+1] = ff0_kernel;

    auto ff1_kernel = new uint32_t [ D_FF* D_MODEL];
    fill_kernel(ff1_kernel, D_FF* D_MODEL);
    weightVec[NUM_HEAD*3 + 2] = ff1_kernel;

    TransformerBlock selfatten(D_SEQ, D_MODEL, D_Q, NUM_HEAD, D_FF, weightVec);
    selfatten.compute(D_SEQ, tensor_in, out, temp);

    std::cout<<"Test finished!!" << std::endl;
}

int main() {
    run_transformer();
    // test_systolic();
    return 0;
}