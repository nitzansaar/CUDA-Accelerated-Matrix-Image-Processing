#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('benchmark_res.csv')

# Calculate mean and std for each N
stats = df.groupby('N')['time_seconds'].agg(['mean', 'std']).reset_index()

# Create the plot
plt.figure(figsize=(12, 6))

# Plot 1: Runtime vs N
plt.subplot(1, 2, 1)
plt.errorbar(stats['N'], stats['mean'], yerr=stats['std'],
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
plt.xlabel('Matrix Size (N)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('CPU Matrix Multiplication Runtime vs Size', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(stats['N'], rotation=45)

plt.tight_layout()
plt.savefig('benchmark_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to: benchmark_plot.png")
plt.show()

# Print statistics table
print("\nBenchmark Statistics:")
print("=" * 50)
print(f"{'N':<10} {'Mean (s)':<15} {'Std Dev (s)':<15}")
