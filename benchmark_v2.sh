#!/usr/bin/env bash
set -euo pipefail

# base size
size=512
runs=2   # number of repeated runs for averaging

# overwrite previous results and write header
echo "impl,N,time_seconds" > benchmark_res.csv

# programs and labels
declare -A programs=(
    ["cpu"]="./matrix_cpu"
    ["naive"]="./matrix_gpu"
    ["tiled"]="./matrix_tiled"
    ["cublas"]="./matrix_cublas"
)

# run benchmarks
for impl in "${!programs[@]}"; do
    exe="${programs[$impl]}"
    echo "=== Benchmarking ${impl^^} Implementation ==="
    for i in 1 2 4; do   # test 512, 1024, 2048
        N=$(( size * i ))
        for run in $(seq 1 $runs); do
            printf "Running %-6s | N=%-5d | Run %d\n" "$impl" "$N" "$run"

            # run the program and capture stdout
            output=$($exe "$N" 2>/dev/null || true)

            # extract numeric time value (the token before 'seconds')
            time_seconds=$(echo "$output" | awk -F": " '{print $2}' | awk '{print $1}')

            # append to CSV
            echo "$impl,$N,$time_seconds" >> benchmark_res.csv

            echo " -> time = ${time_seconds}s"
        done
    done
done

echo
echo "✅ Benchmarking complete. Results saved in benchmark_res.csv"
