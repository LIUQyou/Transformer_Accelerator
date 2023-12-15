#include "transformerBlock.hh"

template<typename T>
TransformerBlock<T>::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads,
                                      std::size_t ff_size, Matrix<T> **weightVector) {

    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    ff_size_ = ff_size;

    for (int i = 0; i < num_heads; ++i) {
        selfatten[i] = new SingleHeadSelfAttn<T>(pre_seq_len, input_dim, head_hidden_size, &weightVector[3*i]);
    }

    condense = new Dense<T>(head_hidden_size * num_heads, input_dim, weightVector[num_heads * 3]);

    multihead_out = new Matrix<T>* [num_heads];
    for (int i = 0; i < num_heads; ++i) {
        multihead_out[i] = new Matrix<T>(pre_seq_len, head_hidden_size);
    }

    condense_out = new Matrix<T>(pre_seq_len, input_dim);
    intermediateFF = new Matrix<T>(pre_seq_len, ff_size);

    addNorm = new AddNormalize<T>(pre_seq_len, input_dim);
    feedForward0 = new Dense<T>(input_dim, ff_size, weightVector[num_heads * 3 + 1]);
    feedForward1 = new Dense<T>(ff_size, input_dim, weightVector[num_heads * 3 + 2]);
}


template<typename T>
TransformerBlock<T>::~TransformerBlock() {
    for (int i = 0; i < num_heads_; ++i) {
        delete selfatten[i];
    }

    for (int i = 0; i < num_heads_; ++i) {
        delete multihead_out[i];
    }
    delete[] multihead_out;

    delete condense;
    delete condense_out;
    delete intermediateFF;
    delete addNorm;
    delete feedForward0;
    delete feedForward1;
}


template<typename T>
void TransformerBlock<T>::compute(std::size_t seq_len, Matrix<T> &input, Matrix<T> &output) {
    // system("m5 resetstats");
    // input [seq_len, input_dim], weight [input_dim, head_hidden_size], output [seq_len, head_hidden_size]
    // seq_len = 512; input_dim = 1024; head_hidden_size = 64; the number of heads = 16
    for (int n=0; n<num_heads_; n++){
        std::cout << "Head : " << n << std::endl;
        selfatten[n]->compute(seq_len, input, *multihead_out[n]);
    }
    // system("m5 dumpresetstats");

    std::cout << "Condense"  << std::endl;
    
    Matrix<T> multihead_out_merge(multihead_out, num_heads_, false);
    // Matrix<T> multihead_out_merge(condense_out->rows_element, condense_out->cols_element);
    // Matrix<T>::merge(multihead_out, num_heads_, multihead_out_merge, false);
    // multihead_out [seq_len, num_heads * head_hidden_size], weight [num_heads * head_hidden_size, input_dim], output [seq_len, input_dim]
    condense->compute(seq_len, multihead_out_merge, *condense_out);
    // system("m5 dumpresetstats");

    // writeToCSV("condense_out.csv", *condense_out);

    std::cout << "Add Norm"  << std::endl;
    // input [seq_len, input_dim], condense_out [seq_len, input_dim], output [seq_len, input_dim]
    addNorm->compute(input, *condense_out);
    // system("m5 dumpresetstats");

    // writeToCSV("condense_out_2.csv", *condense_out);

    std::cout << "Feed Forward 0"  << std::endl;
    // input [seq_len, input_dim], weight [input_dim, ff_size], bias [ff_size], output [seq_len, ff_size]
    feedForward0->compute(seq_len, *condense_out, *intermediateFF);
    
    // system("m5 dumpresetstats");

    std::cout << "Feed Forward 1"  << std::endl;
    // input [seq_len, ff_size], weight [ff_size, input_dim], bias [input_dim], output [seq_len, input_dim]
    feedForward1->compute(seq_len, *intermediateFF, output);
    // system("m5 dumpresetstats");

    std::cout << "Add Norm"  << std::endl;
    // input [seq_len, input_dim], output [seq_len, input_dim]
    addNorm->compute(multihead_out_merge, output);
    // system("m5 dumpresetstats");
    // writeToCSV("input.csv", input);
    // writeToCSV("output.csv", output);

    // // Assuming multihead_out_merge is a merged Matrix representing multihead_out
    // writeToCSV("multihead_out_1.csv", *multihead_out[0]);
    // writeToCSV("multihead_out.csv", multihead_out_merge);
}

template class TransformerBlock<int>;
template class TransformerBlock<float>;
template class TransformerBlock<u_int32_t>;