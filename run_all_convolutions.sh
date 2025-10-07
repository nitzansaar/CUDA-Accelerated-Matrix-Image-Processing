#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------
# Configuration
# -------------------------------------
BINARY=./convolve
OUTPUT_DIR="outputs"
LOG_FILE="benchmark_convolve.csv"

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
  echo "❌ Error: Neither 'convert' nor 'magick' found. Install with:"
  echo "sudo apt-get install -y imagemagick-6.q16"
  exit 1
fi

# -------------------------------------
# Run all combinations
# -------------------------------------
for img in "${IMAGES[@]}"; do
  base="${img%.pgm}"                     # "128x128" from "128x128.pgm"
  mkdir -p "$OUTPUT_DIR/$base"

  for N in "${FILTERS[@]}"; do
    for mode in "${MODES[@]}"; do
      echo "---------------------------------------------"
      echo "Running: $img  (filter=${N}x${N}, mode=$mode)"
      echo "---------------------------------------------"

      # Capture the binary output
      output=$($BINARY "$img" "$base" "$N" "$mode" 2>&1)
      echo "$output"

      # Extract the runtime in seconds (assuming your program prints "time=...s")
      time_sec=$(echo "$output" | grep -oE "[0-9]+\.[0-9]+s" | tr -d 's' | tail -n1)
      if [ -z "$time_sec" ]; then
        # fallback: if printed as "...time=0.12345 s"
        time_sec=$(echo "$output" | grep -oE "[0-9]+\.[0-9]+" | tail -n1)
      fi

      # Save CSV row
      if [ -n "$time_sec" ]; then
        echo "${base},${N},${mode},${time_sec}" >> "$LOG_FILE"
      fi

      # Handle outputs
      pgm_out="output_${mode}_${N}x${N}.pgm"
      if [ -f "$pgm_out" ]; then
        new_pgm="${OUTPUT_DIR}/${base}/${mode}_${N}x${N}.pgm"
        new_jpg="${OUTPUT_DIR}/${base}/${mode}_${N}x${N}.jpg"

        mv "$pgm_out" "$new_pgm"
        echo "✅ Saved: $new_pgm"

        # Convert PGM → JPG
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
# Wrap up
# -------------------------------------
echo "🎉 All convolutions complete!"
echo "Results are in:  $OUTPUT_DIR/"
echo "Benchmark CSV:   $LOG_FILE"
echo
echo "Sample of recorded timings:"
tail -n +2 "$LOG_FILE" | column -t -s,
