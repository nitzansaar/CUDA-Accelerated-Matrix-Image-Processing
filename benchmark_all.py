#!/usr/bin/env python3
import os
import subprocess
import time
import csv
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import pandas as pd
from convolve_wrapper import cuda_convolve, make_filter

# ----------------------------------------
# CONFIG
# ----------------------------------------
IMAGES = ["128x128.pgm", "256x256.pgm", "512x512.pgm", "2048x2048.pgm", "4000x4000.pgm", "8192x8192.pgm", "13000x13000.pgm"]
FILTER_SIZE = 5
MODE = "blur"

CPU_BIN = "./convolve"
GPU_BIN = "./convolve_cuda"
CSV_FILE = "benchmark_by_image.csv"

# ----------------------------------------
# Helper: run a shell binary and extract runtime
# ----------------------------------------
def run_binary(binary, img, base, N, mode):
    cmd = [binary, img, base, str(N), mode]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    output = result.stdout.strip()

    # Try to extract numeric time from output
    for token in output.split():
        if token.replace(".", "", 1).isdigit():
            try:
                val = float(token)
                if 0.00001 < val < 1000:
                    return val
            except:
                continue
    # fallback if parsing fails
    return end - start

# ----------------------------------------
# Helper: run the Python+CUDA library
# ----------------------------------------
def run_python_cuda(img_path, N, mode):
    img = iio.imread(img_path).astype(np.uint8)
    filt = make_filter(N, mode)
    _, t = cuda_convolve(img, filt)
    return t

# ----------------------------------------
# MAIN
# ----------------------------------------
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_size", "filter_size", "mode", "cpu_time", "gpu_time", "python_cuda_time"])

    for img in IMAGES:
        base = img.split(".")[0]
        print(f"\n🧩 Running {img} | {FILTER_SIZE}x{FILTER_SIZE} | {MODE}")

        # CPU
        t_cpu = run_binary(CPU_BIN, img, base, FILTER_SIZE, MODE)
        print(f"CPU: {t_cpu:.6f}s")

        # GPU executable
        t_gpu = run_binary(GPU_BIN, img, base, FILTER_SIZE, MODE)
        print(f"CUDA exe: {t_gpu:.6f}s")

        # Python + CUDA
        t_py = run_python_cuda(img, FILTER_SIZE, MODE)
        print(f"Python+CUDA: {t_py:.6f}s")

        writer.writerow([base, FILTER_SIZE, MODE, t_cpu, t_gpu, t_py])

print("\n✅ Benchmarking complete! Results in:", CSV_FILE)

# ----------------------------------------
# PLOT RESULTS
# ----------------------------------------
df = pd.read_csv(CSV_FILE)
sizes = [int(x.split("x")[0]) for x in df["image_size"]]

plt.figure(figsize=(10,6))
plt.plot(sizes, df["cpu_time"], "o-", label="CPU", linewidth=2, markersize=8)
plt.plot(sizes, df["gpu_time"], "s--", label="CUDA Executable", linewidth=2, markersize=8)
plt.plot(sizes, df["python_cuda_time"], "^--", label="Python + CUDA", linewidth=2, markersize=8)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Image size (pixels per dimension, log scale)")
plt.ylabel("Execution time (s, log scale)")
plt.title(f"Convolution Performance Scaling (Filter={FILTER_SIZE}x{FILTER_SIZE}, Mode={MODE})")
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.savefig("benchmark_by_image.png", dpi=300)
plt.show()

print("\n📊 Comparison chart saved as benchmark_by_image.png")
