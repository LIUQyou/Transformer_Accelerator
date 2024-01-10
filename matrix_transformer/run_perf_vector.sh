#!/bin/bash

# Start and end values for the dimensions
start1=64
end1=2048
stride1=64

start2=3072
end2=8192
stride2=1024

# Loop over the range from start to end, incrementing by stride
for ((dim=$start1; dim<=$end1; dim+=$stride1)); do
    echo "Running test with dimension size: $dim"

    # Custom definitions for the macros
    custom_flags="-DINPUT_DIMENSION=$dim -DOUTPUT_DIMENSION=$dim -DINSIDE_DIMENSION=$dim"

    # Call make with custom CFLAGS_SIZE
    make matrix_test CFLAGS_SIZE="$custom_flags"

    # Run the performance test and store the result in a text file
    ./run_perf3.sh &> "result_O2_${dim}.txt"
done

for ((dim=$start2; dim<=$end2; dim+=$stride2)); do
    echo "Running test with dimension size: $dim"

    # Custom definitions for the macros
    custom_flags="-DINPUT_DIMENSION=$dim -DOUTPUT_DIMENSION=$dim -DINSIDE_DIMENSION=$dim"

    # Call make with custom CFLAGS_SIZE
    make matrix_test CFLAGS_SIZE="$custom_flags"

    # Run the performance test and store the result in a text file
    ./run_perf3.sh &> "result_O2_${dim}.txt"
done

