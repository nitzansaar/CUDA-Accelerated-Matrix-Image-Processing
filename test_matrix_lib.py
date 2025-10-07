import ctypes
import numpy as np
import time

# Load the shared library
lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

# Define argument types for gpu_matrix_multiply
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

# Create matrices
N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Run the GPU multiply
start = time.time()
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
end = time.time()

print(f"Python call to CUDA library completed in {end - start:.4f} seconds")

# Optional verification (compare to NumPy’s matmul)
diff = np.max(np.abs(C - A.dot(B)))
print("Max difference vs NumPy:", diff)
