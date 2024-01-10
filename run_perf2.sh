#!/bin/bash

# Function to profile a program with given perf events
profile_program() {
    program_name=$1
    shift
    events=("$@")

    # Construct the perf stat command with specified events
    perf_cmd="perf stat"
    for event in "${events[@]}"; do
        perf_cmd+=" -e $event"
    done

    # Run the perf stat command with the program
    echo "Running perf stat for $program_name"
    eval "$perf_cmd ./$program_name/transformer"
}

# Events for original_transformer
events_original_transformer=(
    "L1-dcache-loads"
    "L1-dcache-load-misses"
    "L1-dcache-stores"
    "L1-icache-load-misses"
    "LLC-loads"
    "LLC-load-misses"
    "dTLB-loads"
    "dTLB-load-misses"
    "cycles"
)

# Events for matrix_transformer
events_matrix_transformer=(
    "L1-dcache-loads"
    "L1-dcache-load-misses"
    "L1-dcache-stores"
    "L1-icache-load-misses"
    "LLC-loads"
    "LLC-load-misses"
    "dTLB-loads"
    "dTLB-load-misses"
    "cycles"
)

# Profile each program
profile_program original_transformer "${events_original_transformer[@]}"
profile_program matrix_transformer "${events_matrix_transformer[@]}"

