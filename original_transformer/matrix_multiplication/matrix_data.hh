#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>

template<typename T>
class Block {
public:
    static const int BLOCK_WIDTH = 16;
    static const int BLOCK_LENGTH = 64; // 4KB / (16 * 4 bytes) = 64
    T data[BLOCK_WIDTH][BLOCK_LENGTH];

    // Constructor, destructor, and other methods as needed
    static bool isCompatible(const Block& A, const Block& B) {
        return A.BLOCK_WIDTH == B.BLOCK_WIDTH;
    }

    // Matrix multiplication
    static void multiply(const Block& BlockA, const Block& BlockB, Block &result, int offset) {
        // Check if compatible
        if (!isCompatible(BlockA, BlockB)) {
            throw std::invalid_argument("Blocks are not compatible for multiplication.");
        }
        for (int i = 0; i < BLOCK_WIDTH; ++i) {
            for (int j = 0; j < BLOCK_WIDTH; ++j) {
                for (int k = 0; k < BLOCK_LENGTH; ++k) {
                    result.data[i][j + offset*BLOCK_WIDTH] += BlockA.data[i][k] * BlockB.data[j][k];
                }
            }
        }
    }
};

template<typename T>
class Matrix {
public:
    static const int BLOCK_SIZE = 4096; // 4KB
    int rows_block;
    int cols_block;
    int rows_element;
    int cols_element;
    std::vector<std::vector<Block<T>>> blocks;

    Matrix(int rows, int cols);
    ~Matrix();
    // Methods to access and modify matrix elements, etc.
    T& at(int row, int col);

    // print out the data from Matrix by its block index
    void printBlock(int row, int col);

    // print out the data from the whole Matrix
    void printMatrix();

    // Check if two matrices are compatible for multiplication
    static bool isCompatibleForMultiplication(const Matrix& MatrixA, const Matrix& MatrixB) {
        return MatrixA.cols_block == MatrixB.cols_block;
    }

    // Matrix multiplication
    static void multiply(const Matrix& MatrixA, const Matrix& MatrixB, Matrix& result);
};
