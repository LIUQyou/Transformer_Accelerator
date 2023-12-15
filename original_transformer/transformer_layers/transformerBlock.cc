//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"

void print_out(uint32_t* output_array, std::size_t width , std::size_t seq_len){
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < width ; j++)
            std::cout << std::hex << (uint32_t) output_array[i*(width ) + j] << "\t";
        std::cout << std::endl;
    }
}

TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, uint32_t ** weightVector) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    ff_size_ = ff_size;

    for (int n =0; n< num_heads; n++){
        selfatten[n] = new SingleHeadSelfAttn(pre_seq_len, input_dim, head_hidden_size, weightVector+n*3);
    }

    condense = new Dense(num_heads* head_hidden_size, input_dim, weightVector[num_heads * 3]);

    multihead_out = new uint32_t[pre_seq_len * num_heads * head_hidden_size ];
    condense_out = new uint32_t[pre_seq_len * input_dim ];
    intermediateFF = new uint32_t[pre_seq_len * ff_size ];
    intermediateFF_debug = new uint32_t[pre_seq_len * ff_size ];

    addNorm = new AddNormalize(pre_seq_len, input_dim);
    feedForward0 = new Dense(input_dim, ff_size, weightVector[num_heads * 3 + 1]);
    feedForward1 = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2]);
}

TransformerBlock::~TransformerBlock() = default;

void TransformerBlock::compute(std::size_t seq_len, uint32_t *input, uint32_t *output, uint32_t* &temp) {
    // system("m5 resetstats");
    // input [seq_len, input_dim], weight [input_dim, head_hidden_size], output [seq_len, head_hidden_size]
    // seq_len = 512; input_dim = 1024; head_hidden_size = 64; the number of heads = 16
    for (int n=0; n<num_heads_; n++){
        std::cout << "Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, multihead_out + n * (seq_len * head_hidden_size_ ));
    }
    // system("m5 dumpresetstats");
    uint32_t *multihead_out_reshape = new uint32_t[seq_len * num_heads_ * head_hidden_size_ ];
    for (size_t i = 0; i < num_heads_; i++)
    {
        uint32_t *multihead_out_reshape_pointer = multihead_out_reshape + i * head_hidden_size_;
        for (size_t j = 0; j < seq_len*head_hidden_size_; j++)
        {
            multihead_out_reshape_pointer[(j/head_hidden_size_)*head_hidden_size_*num_heads_+j%head_hidden_size_]= multihead_out[i * seq_len * head_hidden_size_ + j];
        }
        
    }
    
    std::cout << "Condense"  << std::endl;
    // multihead_out [seq_len, num_heads * head_hidden_size], weight [num_heads * head_hidden_size, input_dim], output [seq_len, input_dim]
    condense->compute(seq_len, multihead_out_reshape, condense_out);
    // system("m5 dumpresetstats");

    writeToCSV("condense_out.csv", condense_out, 512, seq_len);

    std::cout << "Add Norm"  << std::endl;
    // input [seq_len, input_dim], condense_out [seq_len, input_dim], output [seq_len, input_dim]
    addNorm->compute(input, condense_out);
    // system("m5 dumpresetstats");

    writeToCSV("condense_out_2.csv", condense_out, 512, seq_len);

    std::cout << "Feed Forward 0"  << std::endl;
    // input [seq_len, input_dim], weight [input_dim, ff_size], bias [ff_size], output [seq_len, ff_size]
    feedForward0->compute(seq_len, condense_out, intermediateFF);
    feedForward0->computeSystolic(seq_len, condense_out, intermediateFF_debug);
    for(int i=0; i<seq_len*ff_size_; i++){
        if(intermediateFF[i] != intermediateFF_debug[i]){
            std::cout<<"Error: systolic and tiled results are different"<<std::endl;
            return;
        }
    }
    // system("m5 dumpresetstats");

    std::cout << "Feed Forward 1"  << std::endl;
    // input [seq_len, ff_size], weight [ff_size, input_dim], bias [input_dim], output [seq_len, input_dim]
    feedForward1->compute(seq_len, intermediateFF, output);
    // system("m5 dumpresetstats");

    std::cout << "Add Norm"  << std::endl;
    // input [seq_len, input_dim], output [seq_len, input_dim]
    addNorm->compute(multihead_out_reshape, output);
    // system("m5 dumpresetstats");
    temp = multihead_out_reshape;

    // Writing the arrays to CSV files
    writeToCSV("input.csv", input, 512, seq_len);       // Assuming 'input_dim' is the width for 'input'
    writeToCSV("output.csv", output, 512, seq_len);     // Assuming 'input_dim' is the width for 'output'
    writeToCSV("multihead_out_1.csv", multihead_out_reshape, head_hidden_size_, seq_len);
    writeToCSV("multihead_out.csv", multihead_out_reshape, head_hidden_size_ * num_heads_, seq_len);
}

TransformerBlockSystolic::TransformerBlockSystolic(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size,
                                   std::size_t num_heads, std::size_t ff_size, uint32_t ** weightVector) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    ff_size_ = ff_size;

    for (int n =0; n< num_heads; n++){
        selfatten[n] = new SingleHeadSelfAttnSystolic(pre_seq_len, input_dim, head_hidden_size, weightVector+n*3);
    }

    condense = new Dense(num_heads* head_hidden_size, input_dim, weightVector[num_heads * 3]);

    multihead_out = new uint32_t[pre_seq_len * num_heads * head_hidden_size ];
    condense_out = new uint32_t[pre_seq_len * input_dim ];
    intermediateFF = new uint32_t[pre_seq_len * ff_size ];
    intermediateFF_debug = new uint32_t[pre_seq_len * ff_size ];

    addNorm = new AddNormalize(pre_seq_len, input_dim);
    feedForward0 = new Dense(input_dim, ff_size, weightVector[num_heads * 3 + 1]);
    feedForward1 = new Dense(ff_size, input_dim, weightVector[num_heads * 3 + 2]);
}

TransformerBlockSystolic::~TransformerBlockSystolic() = default;

void TransformerBlockSystolic::compute(std::size_t seq_len, uint32_t *input, uint32_t *output,  uint32_t* &temp) {
    // system("m5 resetstats");
    // input [seq_len, input_dim], weight [input_dim, head_hidden_size], output [seq_len, head_hidden_size]
    // seq_len = 512; input_dim = 1024; head_hidden_size = 64; the number of heads = 16
    for (int n=0; n<num_heads_; n++){
        std::cout << "Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, multihead_out + n * (seq_len * head_hidden_size_ ));
    }
    // system("m5 dumpresetstats");
    uint32_t *multihead_out_reshape = new uint32_t[seq_len * num_heads_ * head_hidden_size_ ];
    for (size_t i = 0; i < num_heads_; i++)
    {
        uint32_t *multihead_out_reshape_pointer = multihead_out_reshape + i * head_hidden_size_;
        for (size_t j = 0; j < seq_len*head_hidden_size_; j++)
        {
            multihead_out_reshape_pointer[(j/head_hidden_size_)*head_hidden_size_*num_heads_+j%head_hidden_size_]= multihead_out[i * seq_len * head_hidden_size_ + j];
        }
        
    }

    std::cout << "Condense"  << std::endl;
    // multihead_out [seq_len, num_heads * head_hidden_size], weight [num_heads * head_hidden_size, input_dim], output [seq_len, input_dim]
    condense->computeSystolic(seq_len, multihead_out_reshape, condense_out);
    // system("m5 dumpresetstats");

    std::cout << "Add Norm"  << std::endl;
    // input [seq_len, input_dim], condense_out [seq_len, input_dim], output [seq_len, input_dim]
    addNorm->compute(input, condense_out);
    // system("m5 dumpresetstats");

    std::cout << "Feed Forward 0"  << std::endl;
    // input [seq_len, input_dim], weight [input_dim, ff_size], bias [ff_size], output [seq_len, ff_size]
    feedForward0->computeSystolic(seq_len, condense_out, intermediateFF);
    feedForward0->compute(seq_len, condense_out, intermediateFF_debug);
    
    for(int i=0; i<seq_len*ff_size_; i++){
        if(intermediateFF[i] != intermediateFF_debug[i]){
            std::cout<<"Error: systolic and tiled results are different"<<std::endl;
            return;
        }
    }
    // system("m5 dumpresetstats");

    std::cout << "Feed Forward 1"  << std::endl;
    // input [seq_len, ff_size], weight [ff_size, input_dim], bias [input_dim], output [seq_len, input_dim]
    feedForward1->computeSystolic(seq_len, intermediateFF, output);
    // system("m5 dumpresetstats");

    std::cout << "Add Norm"  << std::endl;
    // input [seq_len, input_dim], output [seq_len, input_dim]
    addNorm->compute(multihead_out_reshape, output);
    // system("m5 dumpresetstats");
    temp = multihead_out_reshape;
}
