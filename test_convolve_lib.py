import imageio.v3 as iio
import numpy as np
import time
from convolve_wrapper import cuda_convolve, make_filter

# Load a grayscale image
img = iio.imread("512x512.pgm").astype(np.uint8)

# Run CUDA blur
blur_filter = make_filter(3, "blur")
out_blur, t_blur = cuda_convolve(img, blur_filter)
print(f"CUDA blur time: {t_blur:.6f}s")

# Run CUDA edge detection
edge_filter = make_filter(3, "edge")
out_edge, t_edge = cuda_convolve(img, edge_filter)
print(f"CUDA edge time: {t_edge:.6f}s")

# Save outputs
iio.imwrite("python_blur.jpg", out_blur)
iio.imwrite("python_edge.jpg", out_edge)
