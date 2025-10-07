#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------
# Configuration
# -------------------------------------
BINARY=./convolve_cuda
OUTPUT_DIR="outputs_gpu"
LOG_FILE="benchmark_convolve_gpu.csv"

IMAGES=("128x128.pgm" "256x256.pgm" "512x512.pgm")
FILTERS=(3 5 7)
MODES=("blur" "edge")

# -------------------------------------
# Setup
# -------------------------------------
mkdir -p "$OUTPUT_DIR"
echo "image_size,filter_size,mode,time_seconds" > "$LOG_FILE"

# Check that convert or magick exists
if ! command -v convert &>/dev/null && ! command -v magick &>/dev/null; then
  echo "❌ Error: Neither 'convert' nor 'magick' found."
  echo "Install with: sudo apt-get install -y imagemagick-6.q16"
  exit 1
fi

# -------------------------------------
# Run all combinations
# -------------------------------------
for img in "${IMAGES[@]}"; do
  base="${img%.pgm}"   # e.g., 128x128
  mkdir -p "$OUTPUT_DIR/$base"

  for N in "${FILTERS[@]}"; do
    for mode in "${MODES[@]}"; do
      echo "---------------------------------------------"
      echo "Running (GPU): $img | Filter=${N}x${N} | Mode=$mode"
      echo "---------------------------------------------"

      # Run CUDA binary and capture its output
      output=$($BINARY "$img" "$base" "$N" "$mode" 2>&1)
      echo "$output"

      # Extract the timing value from program output
      time_sec=$(echo "$output" | grep -oE "[0-9]+\.[0-9]+s" | tr -d 's' | tail -n1)
      if [ -z "$time_sec" ]; then
        time_sec=$(echo "$output" | grep -oE "[0-9]+\.[0-9]+" | tail -n1)
      fi

      # Append to CSV
      if [ -n "$time_sec" ]; then
        echo "${base},${N},${mode},${time_sec}" >> "$LOG_FILE"
      fi

      # Handle output image files
      pgm_out="output_${mode}_${N}x${N}.pgm"
      if [ -f "$pgm_out" ]; then
        new_pgm="${OUTPUT_DIR}/${base}/${mode}_${N}x${N}.pgm"
        new_jpg="${OUTPUT_DIR}/${base}/${mode}_${N}x${N}.jpg"

        mv "$pgm_out" "$new_pgm"
        echo "✅ Saved: $new_pgm"

        # Convert to JPG
        if command -v convert &>/dev/null; then
          convert "$new_pgm" "$new_jpg"
        else
          magick "$new_pgm" "$new_jpg"
        fi
        echo "✅ Converted: $new_jpg"
      else
        echo "⚠️ No PGM output found for $img (N=$N, mode=$mode)"
      fi
      echo
    done
  done
done

# -------------------------------------
# Done
# -------------------------------------
echo "🎉 All GPU convolutions complete!"
echo "Results:        $OUTPUT_DIR/"
echo "Benchmark CSV:  $LOG_FILE"
echo
echo "Sample timing results:"
tail -n +2 "$LOG_FILE" | column -t -s,
