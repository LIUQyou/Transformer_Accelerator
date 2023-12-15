#pragma once
#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "matrix_data.hh"

//function to print the tensor_in and tensor_out
template<typename T>
void print_tensor(Matrix<T>& tensor_in)
{
    std::cout << "tensor_in: " << std::endl;
    for(std::size_t i = 0; i < tensor_in.rows_element; i++) {
        for(std::size_t j = 0; j < tensor_in.cols_element; j++) {
            std::cout << tensor_in.at(i,j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
void writeToCSV(const std::string& filename, Matrix<T>& matrix) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    for (std::size_t i = 0; i < matrix.rows_element; ++i) {
        for (std::size_t j = 0; j < matrix.cols_element; ++j) {
            file << matrix.at(i, j);
            if (j < matrix.cols_element - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

template<typename T>
bool readFromCSV(const std::string& filename, Matrix<T>& matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    std::size_t row = 0;

    while (getline(file, line) && row < matrix.rows_element) {
        std::stringstream linestream(line);
        std::string cell;
        std::size_t col = 0;

        while (getline(linestream, cell, ',') && col < matrix.cols_element) {
            std::istringstream cellStream(cell);
            T value;
            cellStream >> value;
            matrix.at(row, col) = value;
            col++;
        }
        row++;
    }

    file.close();
    return true;
}

