#!/usr/bin/env bash
set -euo pipefail

# base size
size=256

# overwrite previous results and write header
echo "N,time_seconds" > benchmark_res.csv

# run 5x5 experiments (outer loop name 'run' to avoid reusing the same var)
for run in {1..5}; do
    for i in {1..5}; do
        N=$(( size * i ))
        printf "Running benchmark for N=%d (run %d)\n" "$N" "$run"

        # run the program and capture its stdout
        output=$(./matrix_cpu "$N")

        # matrix_cpu prints: "CPU execution time (N=%d): %f seconds"
        # extract the numeric time (the token immediately before the word 'seconds')
        time_seconds=$(echo "$output" | awk -F": " '{print $2}' | awk '{print $1}')

        # append to CSV
        echo "$N,$time_seconds" >> benchmark_res.csv

        # brief console feedback
        echo "Run $run i=$i -> N=$N time=${time_seconds}s"
    done
done


echo "Benchmarking complete. Results saved in benchmark_res.csv"






