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

    blocks.resize(rows_block);
    for (auto& row : blocks) {
        row.resize(cols_block);
    }

    // Initialize each block, if necessary
}

template<typename T>
Matrix<T>::~Matrix() {}
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
    // The first loop is the rows_block of MatrixA, while the second loop is the rows_block of MatrixB
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
    int rows = blocks.size() * Block<T>::BLOCK_WIDTH;
    int cols = blocks[0].size() * Block<T>::BLOCK_LENGTH;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

template class Matrix<int>;
template class Matrix<float>;