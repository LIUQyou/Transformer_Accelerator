#!/bin/bash

# Start and end values for the dimensions
start=1024
end=8192
stride=1024

# Loop over the range from start to end, incrementing by stride
for ((dim=$start; dim<=$end; dim+=$stride)); do
    echo "Running test with dimension size: $dim"

    # Custom definitions for the macros
    custom_flags="-DINPUT_DIMENSION=$dim -DOUTPUT_DIMENSION=$dim -DINSIDE_DIMENSION=$dim"

    # Call make with custom CFLAGS_SIZE
    make eigen_matrix CFLAGS_SIZE="$custom_flags"

    # Run the performance test and store the result in a text file
    ./run_perf4.sh &> "result_eigen_${dim}.txt"
done

