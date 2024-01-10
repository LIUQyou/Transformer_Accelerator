#include "matrix_data.hh"
//#define INPUT_DIMENSION 256
//#define OUTPUT_DIMENSION 64
//#define INSIDE_DIMENSION 512

typedef u_int32_t Data_Type;

static void block_block_multiplication(Data_Type* A, Data_Type* B, Data_Type* C)
{
    int i, j, k;
    for(i = 0; i < Block<int>::BLOCK_WIDTH; i++) {
        for(j = 0; j < Block<int>::BLOCK_WIDTH; j++) {
            Data_Type sum = 0;
            for(k = 0; k < Block<int>::BLOCK_LENGTH; k++) {
                sum += A[i * Block<int>::BLOCK_LENGTH + k] * B[k * Block<int>::BLOCK_WIDTH + j];
            }
            C[i * Block<int>::BLOCK_WIDTH + j] = sum;
        }
    }
}

// The input A is [rows_matrixA, inside_dimension], B is [inside_dimension, cols_matrixB], C is [rows_matrixA, cols_matrixB]
static void matrix_matrix_multiplication(Data_Type* A, Data_Type* B, Data_Type* C, int rows_matrixA, int inside_dimension, int cols_matrixB)
{
    int i, j, k;
    for(i = 0; i < rows_matrixA; i++) {
        for(j = 0; j < cols_matrixB; j++) {
            Data_Type sum = 0;
            for(k = 0; k < inside_dimension; k++) {
                sum += A[i * inside_dimension + k] * B[k * cols_matrixB + j];
            }
            C[i * cols_matrixB + j] = sum;
        }
    }
}

int block_block_test() {
    Block<int> blockA, blockB;
    Data_Type* A = new Data_Type[Block<int>::BLOCK_WIDTH * Block<int>::BLOCK_LENGTH];
    Data_Type* B = new Data_Type[Block<int>::BLOCK_LENGTH * Block<int>::BLOCK_WIDTH];
    // Initialize blocks with test data
    for (int i = 0; i < Block<int>::BLOCK_WIDTH; ++i) {
        for (int j = 0; j < Block<int>::BLOCK_LENGTH; ++j) {
            blockA.data[i][j] = rand()%5; // or some other test values
            blockB.data[i][j] = rand()%5; // or some other test values
            A[i * Block<int>::BLOCK_LENGTH + j] = blockA.data[i][j];
            B[j * Block<int>::BLOCK_WIDTH + i] = blockB.data[i][j];
        }
    }

    // Result block
    Block<int> result;
    Data_Type* C = new Data_Type[Block<int>::BLOCK_WIDTH * Block<int>::BLOCK_WIDTH];
    // Initialize result block to zero
    for (int i = 0; i < Block<int>::BLOCK_WIDTH; ++i) {
        for (int j = 0; j < Block<int>::BLOCK_WIDTH; ++j) {
            result.data[i][j] = 0;
        }
    }

    // Perform multiplication
    Block<int>::multiply(blockA, blockB, result, 0);
    block_block_multiplication(A, B, C);
    // Print result
    // std::cout << "Result of Block<int> Multiplication:" << std::endl;

    // compare the reuslt from result and C
    for (int i = 0; i < Block<int>::BLOCK_WIDTH; ++i) {
        for (int j = 0; j < Block<int>::BLOCK_WIDTH; ++j) {
            if (result.data[i][j] != C[i * Block<int>::BLOCK_WIDTH + j]) {
                std::cout << "Error: result.data[" << i << "][" << j << "] = " << result.data[i][j] <<
                 ", C[" << i << "][" << j << "] = " << C[i * Block<int>::BLOCK_WIDTH + j] << std::endl;
            }
        }
    }
    // print out the data from blockA, blockB, result
    // std::cout << "Block<int> A:" << std::endl;
    // for (int i = 0; i < Block<int>::BLOCK_WIDTH; ++i) {
    //     for (int j = 0; j < Block<int>::BLOCK_LENGTH; ++j) {
    //         std::cout << blockA.data[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Block<int> B:" << std::endl;
    // for (int i = 0; i < Block<int>::BLOCK_WIDTH; ++i) {
    //     for (int j = 0; j < Block<int>::BLOCK_LENGTH; ++j) {
    //         std::cout << blockB.data[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < Block<int>::BLOCK_WIDTH; ++i) {
    //     for (int j = 0; j < Block<int>::BLOCK_WIDTH; ++j) {
    //         std::cout << result.data[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}

int matrix_matrix_test()
{
    Matrix<int> matrixA(INPUT_DIMENSION, INSIDE_DIMENSION);
    Matrix<int> matrixB(OUTPUT_DIMENSION, INSIDE_DIMENSION);
    Data_Type* matrixA_ = new Data_Type[INPUT_DIMENSION * INSIDE_DIMENSION];
    Data_Type* matrixB_ = new Data_Type[INSIDE_DIMENSION * OUTPUT_DIMENSION];
    // Initialize blocks with test data
    for (int i = 0; i < INPUT_DIMENSION; ++i) {
        for (int j = 0; j < INSIDE_DIMENSION; ++j) {
            matrixA.at(i, j) = rand()%5; // or some other test values
            matrixA_[i * INSIDE_DIMENSION + j] = matrixA.at(i, j);
        }
    }
    for (int i = 0; i < OUTPUT_DIMENSION; ++i) {
        for (int j = 0; j < INSIDE_DIMENSION; ++j) {
            matrixB.at(i, j) = rand()%5; // or some other test values
            matrixB_[j * OUTPUT_DIMENSION + i] = matrixB.at(i, j);
        }
    }

    // Result block
    Matrix<int> result(INPUT_DIMENSION, OUTPUT_DIMENSION);
    Data_Type* result_ = new Data_Type[INPUT_DIMENSION * OUTPUT_DIMENSION];
    // Initialize result block to zero
    for (int i = 0; i < INPUT_DIMENSION; ++i) {
        for (int j = 0; j < OUTPUT_DIMENSION; ++j) {
            result.at(i, j) = 0;
            result_[i * OUTPUT_DIMENSION + j] = 0;
        }
    }

    // Perform multiplication
    Matrix<int>::multiply(matrixA, matrixB, result);
    matrix_matrix_multiplication(matrixA_, matrixB_, result_, INPUT_DIMENSION, INSIDE_DIMENSION, OUTPUT_DIMENSION);

    // compare the reuslt from result and result_
    for (int i = 0; i < INPUT_DIMENSION; ++i) {
        for (int j = 0; j < OUTPUT_DIMENSION; ++j) {
            if (result.at(i, j) != result_[i * OUTPUT_DIMENSION + j]) {
                std::cout << "Error: result.data[" << i << "][" << j << "] = " << result.at(i, j) <<
                 ", result_[" << i << "][" << j << "] = " << result_[i * OUTPUT_DIMENSION + j] << std::endl;
            }
        }
    }

    delete[] matrixA_;
    delete[] matrixB_;
    delete[] result_;

    return 0;
}

int matrix_matrix_original()
{
    // Matrix<int> matrixA(INPUT_DIMENSION, INSIDE_DIMENSION);
    // Matrix<int> matrixB(OUTPUT_DIMENSION, INSIDE_DIMENSION);
    Data_Type* matrixA_ = new Data_Type[INPUT_DIMENSION * INSIDE_DIMENSION];
    Data_Type* matrixB_ = new Data_Type[INSIDE_DIMENSION * OUTPUT_DIMENSION];
    // Initialize blocks with test data
    for (int i = 0; i < INPUT_DIMENSION; ++i) {
        for (int j = 0; j < INSIDE_DIMENSION; ++j) {
            // matrixA.at(i, j) = rand()%5; // or some other test values
            matrixA_[i * INSIDE_DIMENSION + j] = rand()%5;
        }
    }
    for (int i = 0; i < OUTPUT_DIMENSION; ++i) {
        for (int j = 0; j < INSIDE_DIMENSION; ++j) {
            // matrixB.at(i, j) = rand()%5; // or some other test values
            matrixB_[j * OUTPUT_DIMENSION + i] = rand()%5;
        }
    }

    // Result block
    // Matrix<int> result(INPUT_DIMENSION, OUTPUT_DIMENSION);
    Data_Type* result_ = new Data_Type[INPUT_DIMENSION * OUTPUT_DIMENSION];
    // Initialize result block to zero
    // for (int i = 0; i < INPUT_DIMENSION; ++i) {
    //     for (int j = 0; j < OUTPUT_DIMENSION; ++j) {
    //         result.at(i, j) = 0;
    //         result_[i * OUTPUT_DIMENSION + j] = 0;
    //     }
    // }

    // Perform multiplication
    // Matrix<int>::multiply(matrixA, matrixB, result);
    matrix_matrix_multiplication(matrixA_, matrixB_, result_, INPUT_DIMENSION, INSIDE_DIMENSION, OUTPUT_DIMENSION);

    delete[] matrixA_;
    delete[] matrixB_;
    delete[] result_;

    return 0;
}

int matrix_matrix_Block()
{
    Matrix<int> matrixA(INPUT_DIMENSION, INSIDE_DIMENSION);
    Matrix<int> matrixB(OUTPUT_DIMENSION, INSIDE_DIMENSION);
    // Data_Type* matrixA_ = new Data_Type[INPUT_DIMENSION * INSIDE_DIMENSION];
    // Data_Type* matrixB_ = new Data_Type[INSIDE_DIMENSION * OUTPUT_DIMENSION];
    // Initialize blocks with test data
    for (int i = 0; i < INPUT_DIMENSION; ++i) {
        for (int j = 0; j < INSIDE_DIMENSION; ++j) {
            matrixA.at(i, j) = rand()%5; // or some other test values
            // matrixA_[i * INSIDE_DIMENSION + j] = matrixA.at(i, j);
        }
    }
    for (int i = 0; i < OUTPUT_DIMENSION; ++i) {
        for (int j = 0; j < INSIDE_DIMENSION; ++j) {
            matrixB.at(i, j) = rand()%5; // or some other test values
            // matrixB_[j * OUTPUT_DIMENSION + i] = matrixB.at(i, j);
        }
    }

    // Result block
    Matrix<int> result(INPUT_DIMENSION, OUTPUT_DIMENSION);
    // Data_Type* result_ = new Data_Type[INPUT_DIMENSION * OUTPUT_DIMENSION];
    // Initialize result block to zero
    // for (int i = 0; i < INPUT_DIMENSION; ++i) {
    //     for (int j = 0; j < OUTPUT_DIMENSION; ++j) {
    //         result.at(i, j) = 0;
    //         result_[i * OUTPUT_DIMENSION + j] = 0;
    //     }
    // }

    // Perform multiplication
    Matrix<int>::multiply(matrixA, matrixB, result);
    // matrix_matrix_multiplication(matrixA_, matrixB_, result_, INPUT_DIMENSION, INSIDE_DIMENSION, OUTPUT_DIMENSION);

    // compare the reuslt from result and result_
    // for (int i = 0; i < INPUT_DIMENSION; ++i) {
    //     for (int j = 0; j < OUTPUT_DIMENSION; ++j) {
    //         if (result.at(i, j) != result_[i * OUTPUT_DIMENSION + j]) {
    //             std::cout << "Error: result.data[" << i << "][" << j << "] = " << result.at(i, j) <<
    //              ", result_[" << i << "][" << j << "] = " << result_[i * OUTPUT_DIMENSION + j] << std::endl;
    //         }
    //     }
    // }

    // delete[] matrixA_;
    // delete[] matrixB_;
    // delete[] result_;

    return 0;
}

int main()
{
    // block_block_test();
    // matrix_matrix_test();
#ifdef MATRIX_MATRIX_ORIGINAL
    matrix_matrix_original();
#endif

#ifdef MATRIX_MATRIX_BLOCK
    matrix_matrix_Block();
#endif
    return 0;
}
