CC = g++
CFLAGS = -Wall -Wextra -g -Wno-sign-compare -std=c++11

# Add the .cc files here
SOURCES = matrix_data.cc \
		  matrix_multiplication.cc

matrix_test: $(SOURCES)
	$(CC) $(CFLAGS) -o matrix_test $(SOURCES)

.PHONY: clean
clean:
	rm -f matrix_test
