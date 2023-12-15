CC = g++
CFLAGS = -Wall -Wextra -g -Wno-sign-compare -std=c++11

# Add the .cc files here
SOURCES = matrix_data.cc \
		  dense.cc \
		  softmax.cc \
		  selfattention.cc \
		  addNorm.cc \
		  transformerBlock.cc

.PHONY: matrix_test
matrix_test: $(SOURCES) matrix_multiplication.cc
	$(CC) $(CFLAGS) -o matrix_test $(SOURCES) matrix_multiplication.cc

.PHONY: transformer
transformer: $(SOURCES) transformer.cc
	$(CC) $(CFLAGS) -o transformer $(SOURCES) transformer.cc

.PHONY: clean
clean:
	rm -f matrix_test, main
