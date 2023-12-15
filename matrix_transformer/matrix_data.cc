#include <vector>
#include <iostream>
#include <stdexcept>
#include "matrix_data.hh"


template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
    rows_element = rows;
    cols_element = cols;
    rows_block = (rows + Block<T>::BLOCK_WIDTH - 1) / Block<T>::BLOCK_WIDTH;
    cols_block = (cols + Block<T>::BLOCK_LENGTH - 1) / Block<T>::BLOCK_LENGTH;

    // blocks.resize(rows_block);
    // for (auto& row : blocks) {
    //     row.resize(cols_block);
    // }
    blocks = new Block<T>*[rows_block];
    for (int i = 0; i < rows_block; ++i) {
        // Allocate memory for an array of Block<T> objects without calling constructors
        void* mem = operator new[](cols_block * sizeof(Block<T>));

        // Cast the void pointer to a Block<T>* pointer
        blocks[i] = static_cast<Block<T>*>(mem);

        // Construct each Block<T> in place with the desired constructor
        for (int j = 0; j < cols_block; ++j) {
            new(&blocks[i][j]) Block<T>();
        }
    }
    // Initialize each block, if necessary
}

template<typename T>
Matrix<T>::Matrix(Matrix<T> **matrix_array, int length, bool vertical) {
    if (length <= 0 || matrix_array == nullptr) {
        throw std::invalid_argument("Invalid input for Matrix constructor.");
    }

    if (vertical) {
        rows_element = 0;
        cols_element = matrix_array[0]->cols_element;
        for (int i = 0; i < length; ++i) {
            rows_element += matrix_array[i]->rows_element;
        }
    } else {
        cols_element = 0;
        rows_element = matrix_array[0]->rows_element;
        for (int i = 0; i < length; ++i) {
            cols_element += matrix_array[i]->cols_element;
        }
    }

    rows_block = (rows_element + Block<T>::BLOCK_WIDTH - 1) / Block<T>::BLOCK_WIDTH;
    cols_block = (cols_element + Block<T>::BLOCK_LENGTH - 1) / Block<T>::BLOCK_LENGTH;

    // blocks.resize(rows_block);
    // for (int i = 0; i < rows_block; ++i) {
    //     blocks[i].resize(cols_block);
    //     // for (int j = 0; j < cols_block; ++j) {
    //     //     blocks[i][j] = Block<T>(true);  // Using the new constructor
    //     // }
    // }
    blocks = new Block<T>*[rows_block];
    for (int i = 0; i < rows_block; ++i) {
        // Allocate memory for an array of Block<T> objects without calling constructors
        void* mem = operator new[](cols_block * sizeof(Block<T>));

        // Cast the void pointer to a Block<T>* pointer
        blocks[i] = static_cast<Block<T>*>(mem);
    }
    // Assigning the blocks from the input matrices to the new matrix
    int currentRow = 0;
    int currentCol = 0;
    for (int i = 0; i < length; ++i) {
        if (vertical) {
            for (int j = 0; j < matrix_array[i]->rows_block; ++j) {
                for (int k = 0; k < matrix_array[i]->cols_block; ++k) {
                    // blocks[currentRow + j][k] = Block<T>(matrix_array[i]->blocks[j][k]);
                    new(&blocks[currentRow + j][k]) Block<T>(matrix_array[i]->blocks[j][k]);
                }
            }
            currentRow += matrix_array[i]->rows_block;
        } else {
            for (int j = 0; j < matrix_array[i]->rows_block; ++j) {
                for (int k = 0; k < matrix_array[i]->cols_block; ++k) {
                    // blocks[j][currentCol + k] = Block<T>(matrix_array[i]->blocks[j][k]);
                    new(&blocks[j][currentCol + k]) Block<T>(matrix_array[i]->blocks[j][k]);
                }
            }
            currentCol += matrix_array[i]->cols_block;
        }
    }
}


template<typename T>
Matrix<T>::~Matrix() {
    // Call the destructor for each Block<T> in place
    for (int i = 0; i < rows_block; ++i) {
        if (blocks[i] != nullptr) {
            for (int j = 0; j < cols_block; ++j) {
                // Manually call the destructor for each Block<T>
                blocks[i][j].~Block<T>();
            }
            // Correctly deallocate memory allocated with operator new[]
            operator delete[](blocks[i]);
            blocks[i] = nullptr; // Set to nullptr after deletion
        }
    }

    // Free the pointer array
    delete[] blocks;
    blocks = nullptr; // Set the main pointer to nullptr
}

// Matrix multiplication

template<typename T>
void Matrix<T>::multiply(const Matrix& MatrixA, const Matrix& MatrixB, Matrix& result) {
    if (!isCompatibleForMultiplication(MatrixA,MatrixB)) {
        throw std::invalid_argument("Matrices are not compatible for multiplication.");
    }

    // Asser the size of MatrixA.rows_element, MatrixA.rows_element and result.cols_element,result.rows_element
    if (MatrixA.rows_element != result.rows_element || MatrixB.rows_element != result.cols_element) {
        throw std::invalid_argument("Matrices are not compatible for multiplication.");
    }

    // calculate the result block by block
    // The first loop is the rows_block of MatrixA, 
    // while the second loop is the rows_block of MatrixB
    for (int i = 0; i < MatrixA.rows_block; ++i) {
        for (int j = 0; j < MatrixB.rows_block; ++j) {
            // The third loop is the cols_block of MatrixA and MatrixB
            for (int k = 0; k < MatrixA.cols_block; ++k) {
                // calculate the result block by block
                Block<T>::multiply(MatrixA.blocks[i][k], MatrixB.blocks[j][k], result.blocks[i][j/4], j%4);
            }
        }
    }
}

// Methods to access and modify matrix elements, etc.
template<typename T>
T& Matrix<T>::at(int row, int col) {
    int blockIdx = row / Block<T>::BLOCK_WIDTH;
    int blockCol = col / Block<T>::BLOCK_LENGTH;
    int inBlockRow = row % Block<T>::BLOCK_WIDTH;
    int inBlockCol = col % Block<T>::BLOCK_LENGTH;

    return blocks[blockIdx][blockCol].data[inBlockRow][inBlockCol];
}

// print out the data from Matrix by its block index
template<typename T>
void Matrix<T>::printBlock(int row, int col) {
    for (int i = 0; i < Block<T>::BLOCK_WIDTH; ++i) {
        for (int j = 0; j < Block<T>::BLOCK_LENGTH; ++j) {
            std::cout << blocks[row][col].data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// print out the data from the whole Matrix
template<typename T>
void Matrix<T>::printMatrix() {
    int rows = rows_block * Block<T>::BLOCK_WIDTH;
    int cols = cols_block * Block<T>::BLOCK_LENGTH;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void Matrix<T>::MatrixCompute(std::size_t seq_len, const Matrix<T>& input, Matrix<T>& output, Matrix<T>& weight,
                         std::size_t input_size_, std::size_t output_size_)
{
    // input shape [seq_len, input_size_]
    // weight shape [input_size_, output_size_]
    // output shape [seq_len, output_size_]

    // check the seq_len, input_size_, output_size_
    if (seq_len != input.rows_element) {
        throw std::invalid_argument("Matrices are not compatible for multiplication.");
    }
    if (input_size_ != weight.cols_element || input_size_ != input.cols_element) {
        throw std::invalid_argument("Matrices are not compatible for multiplication.");
    }
    if (output_size_ != weight.rows_element) {
        throw std::invalid_argument("Matrices are not compatible for multiplication.");
    }

    // calculate the output block by block
    multiply(input, weight, output);
}

template<typename T>
void Matrix<T>::transpose(Matrix<T> &input, Matrix<T> &output)
{
    for (int i=0; i < input.rows_element; i++){
        for (int j=0; j < input.cols_element; j++){
            output.at(j,i) = input.at(i,j);
        }
    }
}


// template<typename T>
// void Matrix<T>::merge(Matrix<T>** input, std::size_t length, Matrix<T> &output, bool vertical) {
//     if (length == 0) return;

//     // Calculate dimensions of the merged matrix
//     int rows = vertical ? input[0]->rows_element * length : input[0]->rows_element;
//     int cols = vertical ? input[0]->cols_element : input[0]->cols_element * length;

//     // Initialize the output matrix with calculated dimensions
//     output = Matrix<T>(rows, cols);

//     for (std::size_t i = 0; i < length; ++i) {
//         if (!input[i]) continue; // Skip null pointers

//         // Calculate the starting point for the current matrix
//         int start_row = vertical ? i * input[i]->rows_block : 0;
//         int start_col = vertical ? 0 : i * input[i]->cols_block;

//         // Iterate through the blocks of the current matrix
//         for (int row = 0; row < input[i]->rows_block; ++row) {
//             for (int col = 0; col < input[i]->cols_block; ++col) {
//                 // Calculate the position in the output matrix
//                 int out_row = start_row + row;
//                 int out_col = start_col + col;

//                 // Share the block pointer without copying the data
//                 output.blocks[out_row][out_col] = input[i]->blocks[row][col];
//             }
//         }
//     }
// }


template class Matrix<int>;
template class Matrix<float>;
template class Matrix<u_int32_t>;