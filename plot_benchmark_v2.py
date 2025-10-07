#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("benchmark_res.csv")

# Group by implementation and N
stats = df.groupby(["impl", "N"])["time_seconds"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(10, 6))

# Define colors and labels
colors = {"cpu": "tab:red", "naive": "tab:blue", "tiled": "tab:green"}

for impl, group in stats.groupby("impl"):
    plt.errorbar(group["N"], group["mean"], yerr=group["std"],
                 label=impl.upper(),
                 marker='o', capsize=5, capthick=2, linewidth=2,
                 color=colors.get(impl, "black"))

plt.xlabel("Matrix Size (N)", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.title("Matrix Multiplication Performance Comparison", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(group["N"])
plt.yscale("log")
plt.tight_layout()

plt.savefig("benchmark_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# Print summary table
print("\nBenchmark Statistics:\n" + "="*60)
print(stats.to_string(index=False))
