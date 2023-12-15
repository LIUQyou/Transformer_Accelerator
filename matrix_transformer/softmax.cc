#include "softmax.hh"
#include <cmath>

static const  uint8_t  lookup[32] = {
        4, 5, 7, 8, 11, 14, 18, 23, 30, 38, 49, 63, 80, 103, 132, 170, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3,
};

template<typename T>
Softmax<T>::Softmax()= default;

template<typename T>
Softmax<T>::~Softmax()= default;

template<typename T>
void Softmax<T>::compute(Matrix<T> &input, std::size_t seq_len){
    for (size_t i = 0; i < seq_len; i++)
    {
        T sum = 0;
        T input_exp[seq_len];
        for (size_t j = 0; j < seq_len; j++)
        {
            // calculate the softmax of the input length
            // input_exp[j] = exp(input.at(i,j)/8);
            input_exp[j] = input.at(i,j)/1024;
            sum += input_exp[j];
        }
        sum = ((int)sum)/seq_len;
        for (size_t j = 0; j < seq_len; j++)
        {
            input.at(i,j) = int(input_exp[j]*256) / sum;
        }
    }
}

// template<typename T>
// void Softmax<T>::post_softmax(Matrix<T> &input, std::size_t seq_len){
//     // for (int i =0; i< seq_len; i++){
//     //     auto* input_ptr = (int8_t*) (input + i * (seq_len >> 2));
//     //     for (int j=0; j< seq_len; j++){
//     //         *input_ptr = (int8_t) (*(input_ptr) >> 6);
//     //         input_ptr++;
//     //     }
//     // }
// }

template class Softmax<int>;
template class Softmax<float>;
template class Softmax<u_int32_t>;
