#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cstdlib> // For posix_memalign
#include <cstring> // For std::memset
#include <unistd.h> // For getpagesize
#include <atomic>

template<typename T>
class Block {
public:
    static const int BLOCK_WIDTH = 16;
    static const int BLOCK_LENGTH = 64; // 4KB / (16 * 4 bytes) = 64
    T** data;
    std::atomic<int>* ref_count; // Reference counter

    // // Default Constructor
    // Block() {
    //     data =new T* [BLOCK_WIDTH];
    //     for (int i = 0; i < BLOCK_WIDTH; ++i) {
    //         data[i] = new T[BLOCK_LENGTH];
    //     }
    // }
    // Default Constructor
    Block() {
        // Allocate continuous memory for all elements
        T* blockMemory;
        posix_memalign(reinterpret_cast<void**>(&blockMemory), getpagesize(), BLOCK_WIDTH * BLOCK_LENGTH * sizeof(T));
        std::memset(blockMemory, 0, BLOCK_WIDTH * BLOCK_LENGTH * sizeof(T));

        // Allocate memory for data pointers
        data = new T*[BLOCK_WIDTH];

        // Point data pointers to the correct positions in blockMemory
        for (int i = 0; i < BLOCK_WIDTH; ++i) {
            data[i] = &blockMemory[i * BLOCK_LENGTH];
        }
        
        // Initialize reference counter
        ref_count = new std::atomic<int>(1);
    }

    // Constructor copy data from other Block
    Block(const Block& otherBlock) : data(otherBlock.data), ref_count(otherBlock.ref_count) {
        // Increment the reference counter
        ++(*ref_count);
    }

    // Destructor
    ~Block() {
        // Decrement the reference counter and only free memory if it's the last reference
        if (ref_count && --(*ref_count) == 0) {
            // Free the continuous memory block
            if (data != nullptr && data[0] != nullptr) {
                free(data[0]);
                data[0] = nullptr;
            }

            // Free the pointer array
            delete[] data;
            data = nullptr;

            // Delete the reference counter
            delete ref_count;
            ref_count = nullptr;
        }
    }

    // Constructor, destructor, and other methods as needed
    static bool isCompatible(const Block& A, const Block& B) {
        return A.BLOCK_WIDTH == B.BLOCK_WIDTH;
    }

    Block& operator=(const Block& other) {
        if (this != &other) { // Protect against self-assignment
            data = other.data; // Share the data
        }
        return *this;
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
template class Block<int>;
template class Block<float>;
template class Block<u_int32_t>;

template<typename T>
class Matrix {
public:
    static const int BLOCK_SIZE = 4096; // 4KB
    int rows_block;
    int cols_block;
    int rows_element;
    int cols_element;
    Block<T> **blocks;

    Matrix(int rows, int cols);
    Matrix(Matrix **matrix_array, int length, bool vertical = true);
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

    
    static void MatrixCompute(std::size_t seq_len, const Matrix& input, Matrix& output, Matrix& weight,
                         std::size_t input_size_, std::size_t output_size_);

    static void transpose(Matrix& input, Matrix& output);

    // Merge several Matrix into one
    // static void merge(Matrix** input, std::size_t length, Matrix &output, bool vertical = true);
};
