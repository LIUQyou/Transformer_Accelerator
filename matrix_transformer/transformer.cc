#include "dense.hh"
#include "softmax.hh"
#include "selfattention.hh"
#include "transformer_define.hh"
#include "matrix_data.hh"
#include "utils.hh"
#include "transformerBlock.hh"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

typedef u_int32_t Data_Type;

template<typename T>
static void Matrix_initialize(Matrix<T> &matrix, bool transpose = false) {
    if (transpose)
    {
        for(std::size_t i = 0; i < matrix.rows_element; i++) {
            for(std::size_t j = 0; j < matrix.cols_element; j++) {
                matrix.at(i,j) = (j*matrix.rows_element+i)%5;
            }
        }
    }
    else
    {
        for(std::size_t i = 0; i < matrix.rows_element; i++) {
            for(std::size_t j = 0; j < matrix.cols_element; j++) {
                matrix.at(i,j) = (j+i*matrix.cols_element)%5;
            }
        }
    }
}

template<typename T>
static void Matrix_initialize_rand(Matrix<T> &matrix) {
    for(std::size_t i = 0; i < matrix.rows_element; i++) {
        for(std::size_t j = 0; j < matrix.cols_element; j++) {
            matrix.at(i,j) = rand()%7;
        }
    }
}

int main()
{
    // instialize the weight matrix array here
    Matrix<Data_Type> *weightDense[3*NUM_HEADS+3];

    // instialize the input tensor here: [batch_size, seq_len, input_dim]
    Matrix<Data_Type> tensor_in(INPUT_DIMENSION, INSIDE_DIMENSION);
    Matrix<Data_Type> tensor_out_partial(INPUT_DIMENSION, OUTPUT_DIMENSION);
    Matrix<Data_Type> tensor_out(INPUT_DIMENSION, INSIDE_DIMENSION);

    Matrix_initialize_rand(tensor_in);
    writeToCSV("tensor_in.csv", tensor_in);

    // instialize the weight matrix array here
    for (size_t i = 0; i < NUM_HEADS; i++)
    {
        auto query_kernel = new Matrix<Data_Type>(OUTPUT_DIMENSION, INSIDE_DIMENSION);
        Matrix_initialize(*query_kernel, true);
        weightDense[i*3] = query_kernel;

        auto key_kernel = new Matrix<Data_Type>(OUTPUT_DIMENSION, INSIDE_DIMENSION);
        Matrix_initialize(*key_kernel, true);
        weightDense[i*3 + 1] = key_kernel;

        auto value_kernel = new Matrix<Data_Type>(OUTPUT_DIMENSION, INSIDE_DIMENSION);
        Matrix_initialize(*value_kernel, true);
        weightDense[i*3 + 2] = value_kernel;
    }
    
    auto condense_kernel = new Matrix<Data_Type>(INSIDE_DIMENSION, OUTPUT_DIMENSION*NUM_HEADS);
    Matrix_initialize(*condense_kernel, true);
    weightDense[NUM_HEADS*3] = condense_kernel;

    auto ff0_kernel = new Matrix<Data_Type>(FF_DIMENSION, INSIDE_DIMENSION);
    Matrix_initialize(*ff0_kernel, true);
    weightDense[NUM_HEADS*3+1] = ff0_kernel;

    auto ff1_kernel = new Matrix<Data_Type>(INSIDE_DIMENSION, FF_DIMENSION);
    Matrix_initialize(*ff1_kernel, true);
    weightDense[NUM_HEADS*3 + 2] = ff1_kernel;

    TransformerBlock<Data_Type> selfatten(INPUT_DIMENSION, INSIDE_DIMENSION, OUTPUT_DIMENSION, NUM_HEADS,FF_DIMENSION, weightDense);

    std::cout << "selfatten start here!" << std::endl;
    selfatten.compute(INPUT_DIMENSION, tensor_in, tensor_out);

    std::cout << "selfatten finished here!" << std::endl;

    // print_tensor(tensor_in);
    // print_tensor(tensor_out);
    
    // delete the weight matrix array here
    for (size_t i = 0; i <= NUM_HEADS; i++)
    {
        delete weightDense[i*3];
        delete weightDense[i*3 + 1];
        delete weightDense[i*3 + 2];
    }

    return 0;
}