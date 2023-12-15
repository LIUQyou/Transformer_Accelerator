#include "addNorm.hh"
#include <cmath>

template<typename T>
AddNormalize<T>::AddNormalize(std::size_t seq_len, std::size_t input_dim) {
    input_dim_ = input_dim;
    seq_len_ = seq_len;
}

template<typename T>
void AddNormalize<T>::compute(Matrix<T> &input, Matrix<T> &output) {
    auto* temp_ptr = new T[input_dim_];
    for (int i =0; i< seq_len_; i++){
        int32_t sum = 0;
        for (int j=0; j< input_dim_; j++){
            temp_ptr[j] = output.at(i,j) + input.at(i,j);
            sum += temp_ptr[j];
        }

        auto mean = (int32_t) (sum / input_dim_);
        int32_t variance = 0;
        for (int j=0; j< input_dim_; j++){
            variance+= (temp_ptr[j] - mean) * (temp_ptr[j] - mean); // Assuming that the values are fixed-point with 2 digit of fraction.
        }
        variance = variance / (int) input_dim_;
        double sd = sqrt((double) variance);
        auto sd_inv = (int32_t) ((1<<2)/(sd + 1)); // prevent zero divide! // Assuming that the values are fixed-point with 2 digit of fraction.

        for (int j=0; j< input_dim_; j++){
            output.at(i,j) = (temp_ptr[j] - mean) * (sd_inv);
        }

    }

    delete[] temp_ptr;
}

template class AddNormalize<int>;
template class AddNormalize<float>;
template class AddNormalize<u_int32_t>;