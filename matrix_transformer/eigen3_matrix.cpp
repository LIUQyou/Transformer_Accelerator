#include <Eigen/Dense>
#include <iostream>

int main() {
    #ifndef INPUT_DIMENSION
    #define INPUT_DIMENSION 1024
    #endif

    #ifndef OUTPUT_DIMENSION
    #define OUTPUT_DIMENSION 1024
    #endif

    #ifndef INSIDE_DIMENSION
    #define INSIDE_DIMENSION 1024
    #endif

    // Define matrices with dimensions specified by the macros
    Eigen::MatrixXf matrix1 = Eigen::MatrixXf::Random(INPUT_DIMENSION, INSIDE_DIMENSION);
    Eigen::MatrixXf matrix2 = Eigen::MatrixXf::Random(INSIDE_DIMENSION, OUTPUT_DIMENSION);

    // Perform matrix multiplication
    Eigen::MatrixXf result = matrix1 * matrix2;

    // Optional: Output the result dimensions
    std::cout << "Result matrix dimensions: " 
              << result.rows() << "x" << result.cols() << std::endl;

    return 0;
}

