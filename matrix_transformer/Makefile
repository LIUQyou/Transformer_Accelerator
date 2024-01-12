CC = g++
CFLAGS = -Wall -Wextra -Wno-sign-compare -std=c++11 -O2 -Wunused-result

CFLAGS_ORIGINAL = -DMATRIX_MATRIX_ORIGINAL
CFLAGS_BLOCK = -DMATRIX_MATRIX_BLOCK
CFLAGS_SIZE ?= -DINPUT_DIMENSION=256 -DOUTPUT_DIMENSION=64 -DINSIDE_DIMENSION=512

# Add the .cc files here
SOURCES = matrix_data.cc \
		  dense.cc \
		  softmax.cc \
		  selfattention.cc \
		  addNorm.cc \
		  transformerBlock.cc

.PHONY: matrix_test
matrix_test: $(SOURCES) matrix_multiplication.cc
	$(CC) $(CFLAGS) $(CFLAGS_SIZE) $(CFLAGS_ORIGINAL) -o matrix_test_original $(SOURCES) matrix_multiplication.cc
	$(CC) $(CFLAGS) $(CFLAGS_SIZE) $(CFLAGS_BLOCK) -o matrix_test_Block $(SOURCES) matrix_multiplication.cc

.PHONY: transformer
transformer: $(SOURCES) transformer.cc
	$(CC) $(CFLAGS) -o transformer $(SOURCES) transformer.cc

.PHONY: eigen_matrix
eigen_matrix: eigen3_matrix.cpp
	$(CC) $(CFLAGS) $(CFLAGS_SIZE) -O2 -I/usr/include/eigen3 -o eigen3_matrix_O2 eigen3_matrix.cpp
	$(CC) $(CFLAGS) $(CFLAGS_SIZE) -I/usr/include/eigen3 -o eigen3_matrix eigen3_matrix.cpp

.PHONY: clean
clean:
	rm -f matrix_test, main
