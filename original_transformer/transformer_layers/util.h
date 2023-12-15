#pragma once

#ifndef UTIL_H
#define UTIL_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>


// Function to write the array to a CSV file
inline void writeToCSV(const std::string& filename, uint32_t* array, std::size_t width, std::size_t height) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    for (std::size_t i = 0; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            file << array[i * width + j];
            if (j < width - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Function to read data from a CSV file and store it in an array
inline bool readFromCSV(const std::string& filename, uint32_t* array, std::size_t width, std::size_t height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    std::size_t row = 0;

    while (getline(file, line) && row < height) {
        std::stringstream linestream(line);
        std::string cell;
        std::size_t col = 0;

        while (getline(linestream, cell, ',') && col < width) {
            std::istringstream cellStream(cell);
            uint32_t value;
            if (!(cellStream >> value)) {
                std::cerr << "Error: Invalid data format in CSV file" << std::endl;
                return false;
            }
            array[row * width + col] = value;
            col++;
        }

        if (col != width) {
            std::cerr << "Error: Inconsistent number of columns in CSV file" << std::endl;
            return false;
        }
        
        row++;
    }

    if (row != height) {
        std::cerr << "Error: Inconsistent number of rows in CSV file" << std::endl;
        return false;
    }

    file.close();
    return true;
}


#endif // UTIL_H